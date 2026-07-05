#!/usr/bin/env python3
"""Build two small paired test datasets from the all_dialogue parquet.

Both are a random sample of the same rows so they isolate a single variable —
whether the CoT format uses ``<answer>`` tags:

* ``<out_prefix>_tags.parquet``   — tagged CoT for Qwen2/2.5:
  ``cot_eval`` system prompt, ``<think>`` generation prefix, target wrapped in
  ``<answer>...</answer>``.
* ``<out_prefix>_notags.parquet`` — tag-free CoT for Qwen3/Gemma:
  ``cot_eval_notags`` system prompt, empty generation prefix, bare target.

The user prompt is re-rendered with the current ``simple`` template so the
test data reflects the current pipeline. Usage::

    python scripts/make_tag_test_datasets.py \
        --src data/merged_all_dialogue_eval_prompt.parquet \
        --out_prefix data/test_all_dialogue --n 512
"""
import argparse
import copy

import pandas as pd

from scripts.prompt_templates import (
    SYSTEM_PROMPT_STYLES,
    PROMPT_STYLES,
)

TAGS_SYS = SYSTEM_PROMPT_STYLES["cot_eval"]
NOTAGS_SYS = SYSTEM_PROMPT_STYLES["cot_eval_notags"]
SIMPLE = PROMPT_STYLES["simple"]


def _render_user(row):
    md = row["metadata"]
    return SIMPLE.format(
        dialogue_history=md["dialogue_history"],
        responding_speaker=md["responding_speaker"],
    )


def _build(row, tagged: bool):
    row = copy.deepcopy(row)
    gt = row["reward_model"]["ground_truth"]
    user = _render_user(row)
    system = TAGS_SYS if tagged else NOTAGS_SYS

    row["raw_system_prompt"] = system
    row["raw_user_prompt"] = user
    row["prompt"] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    row["raw_prompt"] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    if tagged:
        row["generation_prefix"] = "<think>"
        row["response_with_tags"] = f"<answer>{gt}</answer>"
    else:
        row["generation_prefix"] = ""
        row["response_with_tags"] = gt
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/merged_all_dialogue_eval_prompt.parquet")
    ap.add_argument("--out_prefix", default="data/test_all_dialogue")
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_parquet(args.src)
    sample = df.sample(n=min(args.n, len(df)), random_state=args.seed).reset_index(drop=True)
    print(f"Sampled {len(sample)} rows from {args.src}")

    tagged = pd.DataFrame([_build(r, True) for _, r in sample.iterrows()])
    notags = pd.DataFrame([_build(r, False) for _, r in sample.iterrows()])

    tp = f"{args.out_prefix}_tags.parquet"
    np_ = f"{args.out_prefix}_notags.parquet"
    tagged.to_parquet(tp, index=False)
    notags.to_parquet(np_, index=False)
    print(f"Wrote {tp} ({len(tagged)} rows, tagged/cot_eval)")
    print(f"Wrote {np_} ({len(notags)} rows, tag-free/cot_eval_notags)")

    # Sanity
    t0, n0 = tagged.iloc[0], notags.iloc[0]
    assert "<answer>" in t0["raw_system_prompt"] and t0["response_with_tags"].startswith("<answer>")
    assert "<answer>" not in n0["raw_system_prompt"] and "<answer>" not in n0["response_with_tags"]
    assert t0["reward_model"]["ground_truth"] == n0["reward_model"]["ground_truth"]
    print("Sanity checks passed.")


if __name__ == "__main__":
    main()
