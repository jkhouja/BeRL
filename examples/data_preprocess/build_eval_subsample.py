#!/usr/bin/env python
"""Build a stratified, representative subset of the full ToM eval suite.

Samples ``min(N, available)`` rows per ``data_source`` (subtype) with a fixed
seed and concatenates them into a single parquet that can be used as the
default validation set across all experiments. This gives per-subtype accuracy
estimates with a tight CI (~3.5pp at N=300) while running ~5x faster than the
full ~40k-prompt suite.

Every source parquet shares the same schema
(``ability, answer, data_source, extra_info, prompt, question, reward_model, story``),
so the subset is drop-in compatible with the full-suite scorers.

Usage:
    python examples/data_preprocess/build_eval_subsample.py \
        --n_per_subtype 300 --seed 42 \
        --output data/cleaned_tom/eval_subsample_300.parquet
"""
import argparse
import os

import pandas as pd

# Source benchmark parquets (relative to --data_dir). fantom uses the 50pct
# split (all subtypes still have >=300 rows). ullman_perturbed is intentionally
# omitted (only 9 rows -> too noisy to report).
SOURCE_FILES = [
    "ToM_test_HiExTi_hint_v3.parquet",  # tomi, explore_tom, hi_tom
    "fantom_test_50pct.parquet",         # 5 fantom subtypes
    "bigtom_test.parquet",               # 3 bigtom subtypes
    "dyntom_test.parquet",               # 3 dyntom subtypes
    "exploretom_infilled_test.parquet",  # exploretom_infilled
    "mmlu_test.parquet",                 # mmlu guardrail
    "opentom_test.parquet",              # 5 opentom subtypes
    "simpletom_test.parquet",            # 3 simpletom subtypes
    "tombench_test.parquet",             # tombench
    "gsm8k_test.parquet",                # numeric-reasoning guardrail (data_source=gsm8k)
]

# data_source values to drop entirely (too small / not reported).
EXCLUDE_SOURCES = {"ullman_perturbed"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_dir", default="data/cleaned_tom")
    ap.add_argument("--n_per_subtype", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--output", default="data/cleaned_tom/eval_subsample_300.parquet"
    )
    args = ap.parse_args()

    frames = []
    for fn in SOURCE_FILES:
        path = os.path.join(args.data_dir, fn)
        if not os.path.exists(path):
            raise FileNotFoundError(f"missing eval source: {path}")
        frames.append(pd.read_parquet(path))
    full = pd.concat(frames, ignore_index=True)
    full = full[~full["data_source"].isin(EXCLUDE_SOURCES)]

    parts = []
    print(f"Stratified sample: min({args.n_per_subtype}, available) per data_source, seed={args.seed}\n")
    for src, grp in full.groupby("data_source", sort=True):
        n = min(args.n_per_subtype, len(grp))
        sampled = grp.sample(n=n, random_state=args.seed)
        parts.append(sampled)
        print(f"    {src}: {n} / {len(grp)}")

    out = pd.concat(parts, ignore_index=True)
    out = out.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    out.to_parquet(args.output, index=False)
    print(
        f"\n\u2713 Wrote {len(out)} rows across {out['data_source'].nunique()} "
        f"subtypes -> {args.output}"
    )


if __name__ == "__main__":
    main()
