#!/usr/bin/env python
"""Build the Q0-A3 shuffled/mismatched-utterance control dataset.

Takes an existing BeRL behavior-prediction parquet and replaces each row's TARGET
human utterance with a *different* row's target (a derangement guaranteed to change
the text). The conversation context (prompt / raw_user_prompt / metadata) is left
untouched, so the only thing that changes is the (context -> target) pairing.

This is the Q0 causal control "reward = likelihood of the CORRECT human utterance"
vs "reward = likelihood of ANY human utterance": the marginal distribution of target
utterances is identical, only the pairing is broken.

Reward-time consistency: the reward worker only consumes reward_model['ground_truth']
(verl/workers/fsdp_workers.py:777). We additionally carry the matching 'answer_pp'
(baseline PP of that target) and 'response_with_tags' from the SAME donor row so the
target triple stays internally consistent regardless of baseline-subtraction settings.

Usage:
    python scripts/build_shuffled_control.py \
        --in data/dcfg_smoke_mix.parquet \
        --out data/dcfg_smoke_mix_shuffled.parquet \
        --seed 42
"""
import argparse
import copy

import numpy as np
import pandas as pd


def _ground_truth(rm):
    if isinstance(rm, dict):
        return rm.get("ground_truth")
    return None


def build_derangement(texts, seed):
    """Return a permutation perm such that texts[perm[i]] != texts[i] for all i."""
    n = len(texts)
    rng = np.random.default_rng(seed)
    for _attempt in range(1000):
        perm = rng.permutation(n)
        bad = [i for i in range(n) if texts[perm[i]] == texts[i]]
        if not bad:
            return perm
        # Repair collisions by swapping each bad index with a random other index
        # whose swap keeps both differing; retry the whole draw if repair fails.
        ok = True
        for i in bad:
            swapped = False
            for j in rng.permutation(n):
                if j == i:
                    continue
                # swapping perm[i] and perm[j] must make both differ
                if texts[perm[j]] != texts[i] and texts[perm[i]] != texts[j]:
                    perm[i], perm[j] = perm[j], perm[i]
                    swapped = True
                    break
            if not swapped:
                ok = False
                break
        if ok and all(texts[perm[i]] != texts[i] for i in range(n)):
            return perm
    raise RuntimeError("Could not build a text-distinct derangement; check for a "
                       "dominating duplicate target in the corpus.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_parquet(args.inp).reset_index(drop=True)
    n = len(df)
    texts = [_ground_truth(rm) for rm in df["reward_model"]]
    if any(t is None for t in texts):
        raise ValueError("Some rows have no reward_model.ground_truth; cannot shuffle.")

    perm = build_derangement(texts, args.seed)

    new_rm = []
    for i in range(n):
        donor = int(perm[i])
        rm = copy.deepcopy(df["reward_model"].iloc[i])
        rm["ground_truth"] = copy.deepcopy(df["reward_model"].iloc[donor]["ground_truth"])
        new_rm.append(rm)
    df["reward_model"] = new_rm

    # Carry the donor's matching target metadata so the triple stays consistent.
    for col in ("answer_pp", "response_with_tags"):
        if col in df.columns:
            donor_vals = [copy.deepcopy(df[col].iloc[int(perm[i])]) for i in range(n)]
            df[col] = donor_vals

    # Sanity: every target text changed.
    changed = sum(_ground_truth(df["reward_model"].iloc[i]) != texts[i] for i in range(n))
    assert changed == n, f"only {changed}/{n} targets changed"
    df.to_parquet(args.out)
    print(f"[shuffled-control] wrote {args.out} rows={n} (all {changed} targets remapped, "
          f"seed={args.seed})")


if __name__ == "__main__":
    main()
