"""Build the "options-context v2" behaviour dataset: dcfg_smoke_mix with the 4 MC
options injected into the ROLLOUT prompt ONLY, while the reward model scores the
log-likelihood of the real next turn against the ORIGINAL, options-free context.

Why v2 supersedes v1 (dcfg_smoke_mix_optctx):
  v1 injected the 4 options (incl. the verbatim gold turn) into BOTH `prompt` (rollout
  input) AND `raw_prompt` (the field the actor-RM re-renders to score
  log P(gold | context + CoT), fsdp_workers._build_actor_rm_inputs / _switch_chat_template).
  Because the gold turn then sat verbatim in the scoring context, scoring it became an
  autoregressive COPY task: measured length-normalised LL collapsed from ~-2..-4 nats/tok
  (the genuine behaviour-prediction regime, ST01) to ~-0.01..-0.4 nats/tok, saturating the
  power reward at its +40 ceiling for EVERY rollout regardless of the CoT. Within-group
  reward variance -> 0 -> GRPO advantage -> 0 (empirically 63/190 steps had exactly-zero
  advantage) -> no learning.

The fix (v2): decouple the two columns.
  * `prompt`            -> dialogue history + the 4 shuffled candidate turns + "reason
                           about which they said, then predict" (the model reasons over
                           the alternatives before answering).
  * `raw_prompt` /
    `raw_user_prompt`   -> LEFT UNCHANGED = the original options-free behaviour context,
                           so the actor-RM scores log P(gold | clean context + CoT) in the
                           informative -2..-4 nats/tok regime where the CoT actually moves
                           the reward.

This is safe/isolated: `prompt` (tokenised for rollout, rl_dataset.__getitem__) and
`raw_prompt` (read by the reward path) are independent parquet columns; nothing requires
them to match, and `_assert_train_val_prompt_consistency` compares train-`prompt` vs
val-`prompt` only, never prompt-vs-raw_prompt. No code changes; every reward-relevant
field (reward_model.ground_truth, generation_prefix, response_with_tags, answer_pp) is
identical to the ST01 anchor.

Usage:
    /mnt/home/judekhouja/.conda/envs/tom/bin/python scripts/build_conv_optctx_v2_dataset.py \
        --smoke data/dcfg_smoke_mix.parquet \
        --dist  data/conv_mc_build/distractors.jsonl \
        --out   data/dcfg_smoke_mix_optctx_v2.parquet
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
        head = orig_user
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
    ap.add_argument("--out", default="data/dcfg_smoke_mix_optctx_v2.parquet")
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
    prompts = []  # ONLY the rollout prompt is modified; raw_prompt / raw_user_prompt untouched
    for i in range(len(df)):
        r = df.iloc[i]
        ds = dist.get(i)
        speaker = dict(r["metadata"]).get("responding_speaker", "")
        gold = str(dict(r["reward_model"]).get("ground_truth", "")).strip()
        prompt = copy.deepcopy(list(r["prompt"]))

        if not ds or not speaker or not gold:
            skipped.append(i)
            prompts.append(prompt)
            continue

        options = [gold] + [x.strip() for x in ds]
        rng = random.Random(args.seed + i)  # SAME shuffle as build_conv_mc_dataset / v1
        rng.shuffle(options)
        options_block = "\n".join(f"({LETTERS[j]}) {options[j]}" for j in range(4))

        orig_user = prompt[1]["content"]
        new_user = rebuild_user(orig_user, speaker, options_block)
        prompt[1]["content"] = new_user
        n_changed += 1
        prompts.append(prompt)

    # Only the rollout prompt changes; raw_prompt + raw_user_prompt keep the clean,
    # options-free behaviour context so the reward stays in the informative regime.
    df["prompt"] = prompts

    df.to_parquet(args.out)
    print(f"wrote {len(df)} rows -> {args.out} (options injected into rollout prompt of "
          f"{n_changed}, skipped {len(skipped)})")

    # integrity checks
    ex = df.iloc[0]
    ru = [m for m in ex["raw_prompt"] if m.get("role") == "user"][0]["content"]
    goldtxt = str(dict(ex["reward_model"]).get("ground_truth", "")).strip()
    print("\n--- example ROLLOUT prompt (has options) ---\n" + ex["prompt"][1]["content"][:900])
    print("\n--- example RAW_PROMPT user (scoring context, must be options-free) ---\n" + ru[:500])
    print("\n[check] gold verbatim in raw_prompt user context:", goldtxt[:50], "...")
    print("[check] gold PRESENT in raw_prompt (should be False):", goldtxt in ru)
    print("[check] 'candidate next turns' in raw_prompt (should be False):",
          "candidate next turns" in ru)


if __name__ == "__main__":
    main()
