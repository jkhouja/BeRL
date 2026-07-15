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

import numpy as np
import pandas as pd
import yaml

from scripts.prompt_templates import (
    SYSTEM_PROMPT_STYLES,
    DEFAULT_SYSTEM_PROMPT,
    validate_prompt_tag_consistency,
)

# ---------------------------------------------------------------------------
# Source registry – maps source name to (module_path, class_name)
# ---------------------------------------------------------------------------
CONVERTERS = {
    "dailydialog": ("scripts.convert_dailydialog", "DailyDialogConverter"),
    "empathetic_dialogues": ("scripts.convert_empathetic_dialogues", "EmpatheticDialoguesConverter"),
    "theory_of_mind": ("scripts.convert_theory_of_mind", "TheoryOfMindConverter"),
    "conversations_gone_awry": ("scripts.convert_conversations_gone_awry", "ConversationsGoneAwryConverter"),
    "casino": ("scripts.convert_casino", "CasinoConverter"),
    "craigslist_bargain": ("scripts.convert_craigslist_bargain", "CraigslistBargainConverter"),
    "dealornodeal": ("scripts.convert_dealornodeal", "DealOrNoDealConverter"),
    "switchboard": ("scripts.convert_switchboard", "SwitchboardConverter"),
    "diplomacy": ("scripts.convert_diplomacy", "DiplomacyConverter"),
    "persuasionforgood": ("scripts.convert_persuasionforgood", "PersuasionForGoodConverter"),
    "thoughttrace": ("scripts.convert_thoughttrace", "ThoughtTraceConverter"),
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
    "prompt_style": "simple",
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

def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into ``base`` (override wins).

    Dict values are merged key-by-key; every other type (including the
    ``datasets`` list) is replaced wholesale by the override.
    """
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str, _seen: Optional[set] = None) -> dict:
    """Load a YAML pipeline config, resolving an optional ``extends:`` parent.

    ``extends`` may be a single path or a list of paths (resolved relative to
    the including file's directory). Parents are merged first, then this file's
    keys override them via :func:`_deep_merge`. This lets per-domain ``dcfg_*``
    configs inherit shared defaults from ``dcfg_base.yaml`` and only declare
    their own ``datasets`` + overrides.
    """
    path = os.path.abspath(path)
    _seen = _seen or set()
    if path in _seen:
        raise ValueError(f"Circular 'extends' detected at {path}")
    _seen.add(path)

    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    parents = cfg.pop("extends", None)
    if not parents:
        return cfg
    if isinstance(parents, str):
        parents = [parents]

    base_dir = os.path.dirname(path)
    merged: dict = {}
    for parent in parents:
        parent_path = parent if os.path.isabs(parent) else os.path.join(base_dir, parent)
        merged = _deep_merge(merged, load_config(parent_path, _seen))
    return _deep_merge(merged, cfg)


def get_converter(source: str):
    """Import and return the converter class for *source*."""
    if source not in CONVERTERS:
        raise ValueError(
            f"Unknown source '{source}'. Available: {list(CONVERTERS.keys())}"
        )
    module_path, class_name = CONVERTERS[source]
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def build_converter_config(
    dataset_entry: dict, pipeline_cfg: dict, dataset_defaults: Optional[dict] = None
) -> dict:
    """Merge dataset entry with defaults, returning a flat config dict
    suitable for passing to a converter's ``config=`` parameter.

    Precedence (lowest → highest): hardcoded ``DATASET_DEFAULTS`` →
    config-level ``dataset_defaults`` block → pipeline ``seed`` →
    per-dataset entry values.
    """
    cfg: Dict[str, Any] = {}
    cfg.update(DATASET_DEFAULTS)
    if dataset_defaults:
        cfg.update(dataset_defaults)
    # Pipeline-level seed overrides the default
    cfg["seed"] = pipeline_cfg.get("seed", cfg.get("seed", DATASET_DEFAULTS["seed"]))
    # Dataset-level values override everything
    for k, v in dataset_entry.items():
        if k == "source":
            continue
        cfg[k] = v
    _validate_tag_consistency(dataset_entry.get("source", "?"), cfg)
    return cfg


def _validate_tag_consistency(source: str, cfg: Dict[str, Any]) -> None:
    """Fail fast if the <answer>-tag knobs disagree for this dataset entry.

    Resolves the effective system prompt (explicit ``system_prompt`` or the
    ``system_prompt_style`` registry entry) and cross-checks it against
    ``add_response_tags`` / ``generation_prefix``. Raises ValueError so a
    misconfigured run stops before generating a mismatched dataset.
    """
    system_prompt = cfg.get("system_prompt")
    if system_prompt is None:
        system_prompt = SYSTEM_PROMPT_STYLES.get(
            cfg.get("system_prompt_style", "default"), DEFAULT_SYSTEM_PROMPT
        )
    warnings = validate_prompt_tag_consistency(
        system_prompt=system_prompt,
        add_response_tags=cfg.get("add_response_tags", True),
        generation_prefix=cfg.get("generation_prefix", "") or "",
        context=f"source '{source}'",
    )
    if warnings:
        raise ValueError(
            "Inconsistent <answer>-tag configuration:\n  - "
            + "\n  - ".join(warnings)
        )


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
# Turn filtering (surprise sampling + length-matched random control)
# ---------------------------------------------------------------------------

def _response_word_lengths(df: pd.DataFrame) -> np.ndarray:
    """Per-row response length in words (from the reward_model ground truth)."""
    gts = [
        r["ground_truth"] if isinstance(r, dict) else r
        for r in df["reward_model"].tolist()
    ]
    return np.array([len(str(g).split()) for g in gts], dtype=int)


def _length_matched_random(
    df: pd.DataFrame,
    lengths: np.ndarray,
    surprise_pos: np.ndarray,
    n_keep: int,
    seed: int,
    n_bins: int = 10,
) -> np.ndarray:
    """Randomly pick ``n_keep`` rows whose response-length distribution matches
    that of the surprise-selected set (``surprise_pos``), selected *without*
    regard to surprisal — the length-matched random control for the S3 ablation.

    Strategy: bin the full corpus by response-length quantiles, count how many
    surprise-selected rows fall in each bin, then randomly draw that many rows
    from the *full* pool within the same bin. If a bin lacks enough rows, take
    what is available and top up the shortfall with a random draw from the rest,
    so the control always has the same total size as the surprise set.
    """
    rng = np.random.default_rng(seed)
    edges = np.unique(np.quantile(lengths, np.linspace(0.0, 1.0, n_bins + 1)))
    # digitize into bins 0..len(edges)-2 (interior edges only)
    bin_idx = np.digitize(lengths, edges[1:-1])
    surprise_bin_counts = np.bincount(bin_idx[surprise_pos], minlength=len(edges) - 1)

    all_pos = np.arange(len(df))
    chosen: List[int] = []
    for b, cnt in enumerate(surprise_bin_counts):
        if cnt == 0:
            continue
        pool = all_pos[bin_idx == b]
        take = int(min(cnt, len(pool)))
        if take > 0:
            chosen.extend(rng.choice(pool, size=take, replace=False).tolist())

    chosen_arr = np.array(chosen, dtype=int)
    if len(chosen_arr) < n_keep:
        remaining = np.setdiff1d(all_pos, chosen_arr)
        need = min(n_keep - len(chosen_arr), len(remaining))
        if need > 0:
            extra = rng.choice(remaining, size=need, replace=False)
            chosen_arr = np.concatenate([chosen_arr, extra])
    return chosen_arr


def apply_turn_filter(parquet_path: str, filter_cfg: dict) -> None:
    """Filter the merged corpus by *ToM-dependence / surprise* of each human turn.

    Surprisal is derived from the ``answer_pp`` column, which — in *this*
    pipeline (:func:`compute_perplexity` → :func:`_answer_avg_log_prob`,
    "log scale, no exp") — stores the frozen scorer's **average log-probability**
    of the turn (always ``<= 0``; higher/near-0 = more predictable, more-negative
    = higher perplexity = more surprising). We therefore define::

        surprisal = -answer_pp        # higher = more surprising / high baseline-PPL

    and select the *highest-surprisal* turns (= lowest ``answer_pp``). A hard
    guard rejects a parquet whose ``answer_pp`` looks like raw *perplexity*
    (positive values, as written by the legacy ``merge_tom.py`` path) so the
    selection can never silently invert on a differently-encoded column.

    Modes (config key ``turn_filter.mode``):
      * ``off``      — no filtering (control / full mix).
      * ``surprise`` — keep the ``keep_fraction`` of turns with the **highest
        surprisal** (least predictable / info-asymmetric / ToM-dependent).
      * ``randlen``  — keep the same number of turns as ``surprise``, drawn at
        random but *length-matched* to the surprise set's response-length
        distribution (the quantity/length-controlled comparison for ``surprise``).

    ``answer_pp`` must already exist on the parquet (enable ``perplexity``).
    """
    mode = str(filter_cfg.get("mode", "off")).lower()
    if mode == "off":
        return

    keep_fraction = float(filter_cfg.get("keep_fraction", 0.5))
    seed = int(filter_cfg.get("seed", 42))
    df = pd.read_parquet(parquet_path)

    if "answer_pp" not in df.columns:
        raise ValueError(
            "turn_filter requires the 'answer_pp' column. Enable perplexity "
            "computation (perplexity.enabled: true) so surprisal can be scored."
        )
    if not (0.0 < keep_fraction <= 1.0):
        raise ValueError(f"turn_filter.keep_fraction must be in (0,1], got {keep_fraction}")

    answer_pp = df["answer_pp"].to_numpy(dtype=float)
    # Guard: this pipeline's answer_pp is an average LOG-PROB (<= 0). Positive
    # values mean the column holds raw perplexity (legacy merge_tom.py encoding),
    # for which surprisal = +answer_pp, not -answer_pp — fail loudly rather than
    # select the exact opposite (most-predictable) turns.
    if np.nanmax(answer_pp) > 1e-6:
        raise ValueError(
            f"turn_filter expects answer_pp to be avg LOG-PROB (all <= 0), but found "
            f"positive values (max={np.nanmax(answer_pp):.3f}). This column looks like "
            f"raw perplexity (legacy encoding); surprisal selection would invert. "
            f"Rebuild the parquet with the current build_dataset.py perplexity stage."
        )

    # surprisal = -log_prob: higher = less predictable = more surprising / high PPL.
    surprisal = -answer_pp

    n = len(df)
    n_keep = max(1, round(n * keep_fraction))
    lengths = _response_word_lengths(df)

    # Surprise set = highest surprisal (lowest answer_pp) first.
    order = np.argsort(-surprisal, kind="stable")
    surprise_pos = order[:n_keep]

    if mode == "surprise":
        keep_pos = surprise_pos
    elif mode == "randlen":
        keep_pos = _length_matched_random(df, lengths, surprise_pos, n_keep, seed)
    else:
        raise ValueError(f"Unknown turn_filter.mode '{mode}' (off|surprise|randlen).")

    kept = df.iloc[np.sort(keep_pos)].reset_index(drop=True)
    kept.to_parquet(parquet_path, index=False)

    s_len = lengths[surprise_pos]
    k_len = _response_word_lengths(kept)
    print(
        f"\n  ✓ turn_filter '{mode}': kept {len(kept)}/{n} rows "
        f"(keep_fraction={keep_fraction}); surprisal(-log_prob) surprise-set "
        f"mean={surprisal[surprise_pos].mean():.3f} vs corpus {surprisal.mean():.3f}; "
        f"resp-words mean surprise-set={s_len.mean():.1f} / kept={k_len.mean():.1f}"
    )


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
    turn_filter_cfg = cfg.get("turn_filter", {})
    dataset_defaults = cfg.get("dataset_defaults", {})
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
            converter_cfg = build_converter_config(ds_entry, pipeline_cfg, dataset_defaults)
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
    # Step 5b – Turn filtering (surprise / length-matched random / off)
    # ------------------------------------------------------------------
    if str(turn_filter_cfg.get("mode", "off")).lower() != "off":
        apply_turn_filter(output_path, turn_filter_cfg)

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
