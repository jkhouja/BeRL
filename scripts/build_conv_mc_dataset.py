"""Assemble the BeRL 4-way multiple-choice ("pick the real next turn") training parquets.

Combines:
  * data/conv_mc_build/rows.jsonl        (dialogue history + speaker + gold turn)
  * data/conv_mc_build/distractors.jsonl (3 LLM-generated distractors per idx)

into rule-based-MC training parquets whose ``reward_model.ground_truth`` is the
JSON string consumed by ``verl.utils.reward_score.tom_mc.compute_score``. The
``data_source`` contains ``convmc`` which routes to that scorer (see
``verl/trainer/main_ppo.py:_select_rm_score_fn``).

Two versions are produced (identical options/gold, differing only in what the
model must emit and how it is scored):

  * ``letter`` (data_source ``convmc_letter``): the model answers with the option
    LETTER (A/B/C/D); scored via the MC-letter matcher. ``ground_truth.answer`` =
    gold letter.
  * ``text`` (data_source ``convmc_text``): the model reproduces the full TEXT of
    the chosen turn; scored via the option-text matcher (``ground_truth.answer`` =
    "" so letter-matching is skipped and the gold ``answer_text`` is matched, with
    the built-in distractor-substring ambiguity guard).

The system prompt is copied verbatim from the smoke-mix parquet so training and
the baked ``cot_eval`` evals stay perfectly aligned (Qwen2.5 tagged recipe:
<think>/<answer> tags). Options are rendered eval-style ``(A) ... (D) ...`` and
the gold letter is placed by a per-row deterministic shuffle (shared across the
two versions so they are matched pair-for-pair).

Usage::

    /mnt/home/judekhouja/.conda/envs/tom/bin/python scripts/build_conv_mc_dataset.py \
        --smoke data/dcfg_smoke_mix.parquet \
        --rows  data/conv_mc_build/rows.jsonl \
        --dist  data/conv_mc_build/distractors.jsonl \
        --out-letter data/dcfg_conv_mc_letter.parquet \
        --out-text   data/dcfg_conv_mc_text.parquet
"""

from __future__ import annotations

import argparse
import json
import random

import pandas as pd

USER_HEAD = """\
Below is a real conversation between two people.
Based on the conversation history, work out which of the options below is what \
{speaker} actually said next.

Dialogue History:
{history}

{options}

"""

QUESTION_LETTER = """\
Question: Which option did {speaker} actually say next? Answer with the single \
letter (A, B, C, or D) of the correct option."""

QUESTION_TEXT = """\
Question: Which option did {speaker} actually say next? Answer by writing out the \
full text of the correct option exactly as it appears above."""

LETTERS = ["A", "B", "C", "D"]


def build(mode, rows, dist, system_prompt, data_source, seed):
    out_rows = []
    skipped = 0
    for idx, r in rows.items():
        ds = dist.get(idx)
        if not ds:
            skipped += 1
            continue
        gold = r["gold"].strip()
        options = [gold] + [x.strip() for x in ds]
        # per-row deterministic shuffle (shared seed -> identical layout across modes)
        rng = random.Random(seed + idx)
        rng.shuffle(options)
        gold_pos = options.index(gold)
        gold_letter = LETTERS[gold_pos]
        choices = {LETTERS[i]: options[i] for i in range(4)}
        options_block = "\n".join(f"({LETTERS[i]}) {options[i]}" for i in range(4))

        question = (QUESTION_LETTER if mode == "letter" else QUESTION_TEXT).format(
            speaker=r["speaker"]
        )
        user = USER_HEAD.format(
            speaker=r["speaker"], history=r["history"], options=options_block
        ) + question

        if mode == "letter":
            gt_answer = gold_letter
            top_answer = gold_letter
        else:  # text: skip letter-matching, match on answer_text
            gt_answer = ""
            top_answer = gold

        ground_truth = json.dumps(
            {
                "answer": gt_answer,
                "answer_text": gold,
                "question_type": "mc",
                "wrong_answer": "",
                "choices": choices,
            }
        )
        out_rows.append(
            {
                "data_source": data_source,
                "prompt": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user},
                ],
                "ability": "theory_of_mind",
                "reward_model": {"ground_truth": ground_truth, "style": "rule"},
                "extra_info": {"key": "dummy", "idx": idx, "src": r["data_source"],
                               "gold_letter": gold_letter},
                "answer": top_answer,
                "question": f"Which option did {r['speaker']} actually say next?",
                "story": r["history"],
            }
        )
    return pd.DataFrame(out_rows), skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", default="data/dcfg_smoke_mix.parquet")
    ap.add_argument("--rows", default="data/conv_mc_build/rows.jsonl")
    ap.add_argument("--dist", default="data/conv_mc_build/distractors.jsonl")
    ap.add_argument("--out-letter", default="data/dcfg_conv_mc_letter.parquet")
    ap.add_argument("--out-text", default="data/dcfg_conv_mc_text.parquet")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    smoke = pd.read_parquet(args.smoke)
    system_prompt = smoke.iloc[0]["prompt"][0]["content"]

    rows = {}
    with open(args.rows) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                rows[int(r["idx"])] = r

    dist = {}
    with open(args.dist) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            if "distractors" in d and len(d.get("distractors", [])) >= 3:
                dist[int(d["idx"])] = d["distractors"][:3]

    for mode, out, data_source in (
        ("letter", args.out_letter, "convmc_letter"),
        ("text", args.out_text, "convmc_text"),
    ):
        df, skipped = build(mode, rows, dist, system_prompt, data_source, args.seed)
        df.to_parquet(out)
        bal = df["answer"].value_counts().to_dict() if mode == "letter" else "(text)"
        print(f"[{mode}] wrote {len(df)} rows -> {out} "
              f"(skipped {skipped} without distractors); letter-balance: {bal}")


if __name__ == "__main__":
    main()
