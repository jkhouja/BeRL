"""Build the "options-context" behaviour dataset: dcfg_smoke_mix + the 4 MC options
injected into the prompt, but scored with the SAME behaviour (log-likelihood) reward
as the ll_baseline (ST01 anchor).

Identical to dcfg_smoke_mix in every reward-relevant field (reward_model.ground_truth
= the real next turn, generation_prefix, response_with_tags, answer_pp) — the ONLY
change is that each user prompt now shows the 4 candidate turns (the real turn + the
same 3 LLM distractors used by dcfg_conv_mc_*, same per-row shuffle) BEFORE asking the
model to predict the next turn.

Why this is not a trivial copy: the behaviour reward is log P(gold | prompt + model CoT)
and GRPO normalises within each prompt-group (the 16 rollouts share the identical
context, incl. the gold-bearing options), so the gold-in-context is a constant offset
that cancels in the advantage — the options act purely as reasoning material for the CoT.

The options are injected into BOTH `prompt` (rollout input) and `raw_prompt` (the field
the reward model re-renders from, fsdp_workers._switch_chat_template) plus the
`raw_user_prompt` string mirror.

Usage:
    /mnt/home/judekhouja/.conda/envs/tom/bin/python scripts/build_conv_optctx_dataset.py \
        --smoke data/dcfg_smoke_mix.parquet \
        --dist  data/conv_mc_build/distractors.jsonl \
        --out   data/dcfg_smoke_mix_optctx.parquet
"""

from __future__ import annotations

import argparse
import copy
import json
import random

import pandas as pd

LETTERS = ["A", "B", "C", "D"]

OPTIONS_INTRO = "Here are four candidate next turns; exactly one is what {speaker} actually said next:"
REASON_LEAD = "First reason about which candidate {speaker} actually said and why (given their beliefs, desires and intentions), then "


def rebuild_user(orig_user: str, speaker: str, options_block: str) -> str:
    """Insert the options block + reasoning lead just before the trailing
    'Now respond ...' instruction of the original behaviour user prompt."""
    marker = "\n\nNow respond with what"
    idx = orig_user.find(marker)
    if idx == -1:
        # fallback: append at end
        head, tail = orig_user, ""
        return (head + "\n\n" + OPTIONS_INTRO.format(speaker=speaker) + "\n" +
                options_block + "\n\n" + REASON_LEAD.format(speaker=speaker) +
                "respond with what " + speaker + " will say next without including the speaker: prefix.")
    head = orig_user[:idx]
    tail = orig_user[idx + len("\n\nNow "):]  # 'respond with what ... prefix.'
    return (head + "\n\n" + OPTIONS_INTRO.format(speaker=speaker) + "\n" +
            options_block + "\n\n" + REASON_LEAD.format(speaker=speaker) + tail)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", default="data/dcfg_smoke_mix.parquet")
    ap.add_argument("--dist", default="data/conv_mc_build/distractors.jsonl")
    ap.add_argument("--out", default="data/dcfg_smoke_mix_optctx.parquet")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_parquet(args.smoke).reset_index(drop=True)

    dist = {}
    with open(args.dist) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            if "distractors" in d and len(d.get("distractors", [])) >= 3:
                dist[int(d["idx"])] = d["distractors"][:3]

    n_changed = 0
    skipped = []
    prompts, raw_prompts, raw_users = [], [], []
    for i in range(len(df)):
        r = df.iloc[i]
        ds = dist.get(i)
        speaker = dict(r["metadata"]).get("responding_speaker", "")
        gold = str(dict(r["reward_model"]).get("ground_truth", "")).strip()
        prompt = copy.deepcopy(list(r["prompt"]))
        raw_prompt = copy.deepcopy(list(r["raw_prompt"]))
        raw_user = r["raw_user_prompt"]

        if not ds or not speaker or not gold:
            skipped.append(i)
            prompts.append(prompt); raw_prompts.append(raw_prompt); raw_users.append(raw_user)
            continue

        options = [gold] + [x.strip() for x in ds]
        rng = random.Random(args.seed + i)  # same shuffle as build_conv_mc_dataset
        rng.shuffle(options)
        options_block = "\n".join(f"({LETTERS[j]}) {options[j]}" for j in range(4))

        orig_user = prompt[1]["content"]
        new_user = rebuild_user(orig_user, speaker, options_block)
        prompt[1]["content"] = new_user
        # raw_prompt mirrors prompt (system+user); update its user message too
        for m in raw_prompt:
            if m.get("role") == "user":
                m["content"] = new_user
        raw_user = new_user
        n_changed += 1

        prompts.append(prompt); raw_prompts.append(raw_prompt); raw_users.append(raw_user)

    df["prompt"] = prompts
    df["raw_prompt"] = raw_prompts
    df["raw_user_prompt"] = raw_users
    df.to_parquet(args.out)
    print(f"wrote {len(df)} rows -> {args.out} (options injected into {n_changed}, "
          f"skipped {len(skipped)})")
    # show one example
    ex = df.iloc[0]["prompt"][1]["content"]
    print("\n--- example user prompt ---\n" + ex[:1400])


if __name__ == "__main__":
    main()
