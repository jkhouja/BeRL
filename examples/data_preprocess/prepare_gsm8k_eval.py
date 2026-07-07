"""
Prepare GSM8K (openai/gsm8k, subset 'main') as a numeric-reasoning *guardrail* eval.

GSM8K is not a ToM task — it is a grade-school math word-problem benchmark used to
check that ToM / dialogue GRPO training does not degrade general numeric reasoning
(the numeric analogue of the MMLU knowledge guardrail).

HF: openai/gsm8k  (subset 'main', split 'test' = 1319 rows).
Each row: question (str), answer (str, chain-of-thought ending in '#### <number>').

The gold answer is the number after '####'. Rows use the SAME cot_eval system prompt
and row schema as every other BeRL eval parquet (see tom_eval_common), and are scored
by verl.utils.reward_score.gsm8k_eval via data_source == 'gsm8k'.

Output: data/cleaned_tom/gsm8k_test.parquet  (data_source = gsm8k)

Usage:
    python examples/data_preprocess/prepare_gsm8k_eval.py \
        --out data/cleaned_tom/gsm8k_test.parquet --limit 1000
"""

import argparse
import re

from datasets import load_dataset

from tom_eval_common import make_messages, make_ground_truth, make_sample, write_parquet


def _gold_number(answer_field: str) -> str:
    """Extract the final numeric answer from a GSM8K 'answer' string ('#### n')."""
    m = re.search(r"####\s*(-?[0-9.,]+)", answer_field)
    raw = m.group(1) if m else answer_field
    return raw.replace(",", "").replace("$", "").strip()


def build(split="test", limit=None):
    ds = load_dataset("openai/gsm8k", "main", split=split)
    if limit:
        ds = ds.shuffle(seed=0).select(range(min(limit, len(ds))))
    rows = []
    for i, e in enumerate(ds):
        gold = _gold_number(e["answer"])
        gt = make_ground_truth(answer=gold, question_type="exact", answer_text=gold)
        msgs = make_messages(context="", question=e["question"])
        msgs[1]["content"] = msgs[1]["content"].lstrip("\n")
        rows.append(make_sample(
            "gsm8k", msgs, gt,
            question=e["question"], answer_text=gold, story="",
            extra_info={"index": i},
        ))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/cleaned_tom/gsm8k_test.parquet")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=1000,
                    help="cap rows (guardrail check doesn't need the full 1319)")
    args = ap.parse_args()
    rows = build(split=args.split, limit=args.limit)
    write_parquet(rows, args.out)


if __name__ == "__main__":
    main()
