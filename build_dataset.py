#!/usr/bin/env python3
"""
Unified config-driven data generation pipeline.

Usage:
    python build_dataset.py --config pipeline_config.yaml
"""

import argparse
import importlib
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Source registry – maps source name to (module_path, class_name)
# ---------------------------------------------------------------------------
CONVERTERS = {
    "dailydialog": ("scripts.convert_dailydialog", "DailyDialogConverter"),
    "empathetic_dialogues": ("scripts.convert_empathetic_dialogues", "EmpatheticDialoguesConverter"),
    "theory_of_mind": ("scripts.convert_theory_of_mind", "TheoryOfMindConverter"),
}

# ---------------------------------------------------------------------------
# Default values for per-dataset config entries
# ---------------------------------------------------------------------------
DATASET_DEFAULTS: Dict[str, Any] = {
    "split": "train",
    "sample_size": None,
    "min_turns": 4,
    "max_turns": None,
    "min_response_words": 5,
    "max_response_words": None,
    "limit_turn": None,
    "prompt_style": "cot",
    "system_prompt_style": "cot",
    "generation_prefix": "<think>",
    "add_response_tags": True,
    "include_emotion_context": True,
    "include_system_in_prompt": False,
    "response_tag_open": "<answer>",
    "response_tag_close": "</answer>",
    "ability": "conversation_generation",
    "system_prompt": None,
    "user_template": None,
    "seed": 42,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    """Load and return the YAML pipeline config."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_converter(source: str):
    """Import and return the converter class for *source*."""
    if source not in CONVERTERS:
        raise ValueError(
            f"Unknown source '{source}'. Available: {list(CONVERTERS.keys())}"
        )
    module_path, class_name = CONVERTERS[source]
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def build_converter_config(dataset_entry: dict, pipeline_cfg: dict) -> dict:
    """Merge dataset entry with defaults, returning a flat config dict
    suitable for passing to a converter's ``config=`` parameter."""
    cfg: Dict[str, Any] = {}
    cfg.update(DATASET_DEFAULTS)
    # Pipeline-level seed overrides the default
    cfg["seed"] = pipeline_cfg.get("seed", DATASET_DEFAULTS["seed"])
    # Dataset-level values override everything
    for k, v in dataset_entry.items():
        if k == "source":
            continue
        cfg[k] = v
    return cfg


def run_converter(source: str, converter_config: dict, tmp_dir: str) -> str:
    """Instantiate a converter, run it, save to a temp parquet, return path."""
    ConverterClass = get_converter(source)
    converter = ConverterClass(config=converter_config)

    print(f"\n{'='*60}")
    print(f"  Converting: {source}")
    print(f"{'='*60}")

    # Download and convert
    dataset = converter.download_dataset()
    df = converter.convert(dataset)

    # Save intermediate parquet
    out_path = os.path.join(tmp_dir, f"{source}.parquet")
    df.to_parquet(out_path, index=False)
    print(f"  → {len(df)} rows saved to {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Perplexity computation (reuses process_data functions)
# ---------------------------------------------------------------------------

def compute_perplexity(parquet_path: str, model_name: str, batch_size: int) -> None:
    """Load a model and compute answer perplexity for every row in the parquet
    file, updating the ``answer_pp`` column in-place on disk."""
    import torch
    import torch.nn.functional as F
    import transformers as T

    def _logprobs_from_logits(logits, labels, response_mask):
        """Compute log-probs at label positions (from process_data.py)."""
        if logits.dtype in [torch.float32, torch.float64]:
            logits_labels = torch.gather(
                logits, dim=-1, index=labels.unsqueeze(-1)
            ).squeeze(-1)
            logsumexp_values = torch.stack(
                [torch.logsumexp(l, dim=-1) for l in logits]
            )
            logprobs_labels = logits_labels - logsumexp_values
        else:
            logprobs_labels = []
            for row_logits, row_labels in zip(logits, labels):
                row_logprobs = F.log_softmax(row_logits, dim=-1)
                row_logprobs_labels = row_logprobs.gather(
                    dim=-1, index=row_labels.unsqueeze(-1)
                ).squeeze(-1)
                logprobs_labels.append(row_logprobs_labels)
            logprobs_labels = torch.stack(logprobs_labels)
        return logprobs_labels * response_mask

    def _answer_avg_log_prob(labels, logits, response_mask):
        """Compute per-example average log probability (log scale, no exp)."""
        log_prob = _logprobs_from_logits(logits, labels, response_mask)
        log_prob = log_prob.sum(dim=-1)
        log_prob = log_prob / response_mask.sum(-1)
        return log_prob  # negative values; higher = more likely

    print(f"\nComputing perplexity with model: {model_name} (batch_size={batch_size})")

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"  Device: {device}")

    model = T.AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = T.AutoTokenizer.from_pretrained(model_name)
    if not tokenizer.pad_token:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.resize_token_embeddings(len(tokenizer))

    df = pd.read_parquet(parquet_path)
    pp_values: List[Optional[float]] = []

    model.eval()
    with torch.no_grad():
        for start in range(0, len(df), batch_size):
            batch_rows = df.iloc[start : start + batch_size]
            prompts = batch_rows["prompt"].tolist()
            ground_truths = [
                r["ground_truth"] if isinstance(r, dict) else r
                for r in batch_rows["reward_model"].tolist()
            ]

            # Tokenise prompt+ground_truth pairs
            texts = []
            prompt_lens = []
            response_word_counts = []
            for prompt_msgs, gt in zip(prompts, ground_truths):
                prompt_text = tokenizer.apply_chat_template(
                    prompt_msgs, tokenize=False, add_generation_prompt=True
                )
                full_text = prompt_text + gt
                texts.append(full_text)
                prompt_lens.append(
                    len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])
                )
                response_word_counts.append(len(gt.split()))

            encoding = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(device)

            input_ids = encoding["input_ids"]
            attention_mask = encoding["attention_mask"]
            seq_len = input_ids.shape[1]

            # Apply causal LM shift: predict next token from current
            shifted_input_ids = input_ids[:, :-1]
            labels = input_ids[:, 1:]
            shifted_attention_mask = attention_mask[:, :-1]

            # Build response mask on shifted sequence.
            # Match the original NegTOMDataset convention:
            #   response_mask[input_len-1:full_len] = 1  (before shift)
            #   response_mask = response_mask[1:]         (after shift)
            # This gives response_mask[input_len-2:full_len-1] = 1 on shifted seq.
            response_mask = torch.zeros_like(labels, dtype=torch.int)
            for i, pl in enumerate(prompt_lens):
                resp_end = shifted_attention_mask[i].sum().item()  # exclude padding
                response_mask[i, max(pl - 2, 0):resp_end] = 1

            model_outputs = model(shifted_input_ids, attention_mask=shifted_attention_mask)

            batch_dict = {
                "prompt_len": prompt_lens,
                "labels": labels,
                "model_outputs": model_outputs,
                "response_mask": response_mask,
                "response_words": torch.tensor(response_word_counts, device=device),
            }

            pp = _answer_avg_log_prob(labels, model_outputs.logits, response_mask)
            if pp.isnan().any() or pp.isinf().any():
                bad_idx = (pp.isnan() | pp.isinf()).nonzero(as_tuple=True)[0]
                raise ValueError(
                    f"answer_pp has NaN/Inf values at batch indices {bad_idx.tolist()} "
                    f"(global rows {start + bad_idx.cpu().tolist()[0]}+). "
                    f"Check input data for empty responses or malformed prompts."
                )
            pp_values.extend(pp.cpu().tolist())

            processed = min(start + batch_size, len(df))
            print(f"  Perplexity: {processed}/{len(df)} rows", end="\r")

    print()
    df["answer_pp"] = pp_values[: len(df)]
    df.to_parquet(parquet_path, index=False)
    print(f"  ✓ answer_pp column written ({len(df)} rows)")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build a training-ready parquet dataset from a YAML config."
    )
    parser.add_argument(
        "--config", type=str, default="scripts/configs/pipeline_config.yaml",
        help="Path to the pipeline YAML config file.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    pipeline_cfg = cfg.get("pipeline", {})
    perplexity_cfg = cfg.get("perplexity", {})
    datasets_cfg: List[dict] = cfg.get("datasets", [])

    if not datasets_cfg:
        print("No datasets specified in config. Nothing to do.")
        sys.exit(0)

    output_dir = pipeline_cfg.get("output_dir", "data")
    output_name = pipeline_cfg.get("output_name", "merged_dataset")
    seed = pipeline_cfg.get("seed", 42)
    shuffle = pipeline_cfg.get("shuffle", True)

    output_path = os.path.join(output_dir, f"{output_name}.parquet")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    # ------------------------------------------------------------------
    # Step 1 – Convert each dataset source
    # ------------------------------------------------------------------
    intermediate_paths: List[str] = []
    with tempfile.TemporaryDirectory(prefix="pipeline_") as tmp_dir:
        for ds_entry in datasets_cfg:
            source = ds_entry.get("source")
            if not source:
                print("WARNING: dataset entry missing 'source', skipping.")
                continue
            converter_cfg = build_converter_config(ds_entry, pipeline_cfg)
            path = run_converter(source, converter_cfg, tmp_dir)
            intermediate_paths.append(path)

        # ------------------------------------------------------------------
        # Step 2 – Merge intermediates
        # ------------------------------------------------------------------
        if len(intermediate_paths) == 1:
            # No merge needed – just copy
            import shutil
            shutil.copy2(intermediate_paths[0], output_path)
        else:
            from merge_parquet import merge_parquet_files
            print(f"\n{'='*60}")
            print("  Merging datasets")
            print(f"{'='*60}")
            merge_parquet_files(intermediate_paths, output_path)

    # ------------------------------------------------------------------
    # Step 3 – Shuffle
    # ------------------------------------------------------------------
    if shuffle:
        print(f"\nShuffling with seed={seed} ...")
        df = pd.read_parquet(output_path)
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
        df.to_parquet(output_path, index=False)
        print(f"  ✓ Shuffled {len(df)} rows")

    # ------------------------------------------------------------------
    # Step 4 – Perplexity (optional)
    # ------------------------------------------------------------------
    if perplexity_cfg.get("enabled", False):
        compute_perplexity(
            output_path,
            model_name=perplexity_cfg.get("model_name", "Qwen/Qwen2.5-3B-Instruct"),
            batch_size=perplexity_cfg.get("batch_size", 8),
        )

    # ------------------------------------------------------------------
    # Step 5 – Validate answer_pp
    # ------------------------------------------------------------------
    df = pd.read_parquet(output_path)
    if "answer_pp" in df.columns:
        null_count = df["answer_pp"].isna().sum()
        if null_count > 0:
            if perplexity_cfg.get("enabled", False):
                raise ValueError(
                    f"answer_pp has {null_count}/{len(df)} null values after "
                    f"perplexity computation. Something went wrong."
                )
            else:
                raise ValueError(
                    f"answer_pp has {null_count}/{len(df)} null values. "
                    f"Enable perplexity computation in the config "
                    f"(perplexity.enabled: true) or pre-compute answer_pp "
                    f"before training."
                )

    # ------------------------------------------------------------------
    # Step 6 – Summary
    # ------------------------------------------------------------------
    elapsed = time.time() - start_time
    df = pd.read_parquet(output_path)
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)

    print(f"\n{'='*60}")
    print("  Pipeline complete!")
    print(f"{'='*60}")
    print(f"  Output : {output_path}")
    print(f"  Rows   : {len(df)}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Size   : {file_size_mb:.2f} MB")
    print(f"  Time   : {elapsed:.1f}s")

    if "data_source" in df.columns:
        print("\n  Per-source counts:")
        for src, count in df["data_source"].value_counts().items():
            print(f"    {src}: {count}")

    print()


if __name__ == "__main__":
    main()
