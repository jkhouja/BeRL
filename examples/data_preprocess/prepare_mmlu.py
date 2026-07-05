"""
Prepare MMLU (cais/mmlu, subset 'all') as a *guardrail* eval for BeRL.

MMLU is not a ToM task — it's a general-knowledge 4-way MC benchmark used to check
that ToM/dialogue GRPO training does not degrade broad reasoning/knowledge.

HF: cais/mmlu  (subset 'all', split 'test' = 14042 rows)
Each row: question, subject, choices (list[str] len 4), answer (int 0-3).

Output: data/cleaned_tom/mmlu_test.parquet  (data_source = mmlu)

Usage:
    python examples/data_preprocess/prepare_mmlu.py \
        --out data/cleaned_tom/mmlu_test.parquet --limit 2000
"""

import argparse

from datasets import load_dataset

from tom_eval_common import (
    make_messages, format_choices, make_ground_truth, make_sample, write_parquet,
)


def build(split="test", limit=None, shuffle=True):
    ds = load_dataset("cais/mmlu", "all", split=split)
    if limit:
        ds = ds.shuffle(seed=0).select(range(min(limit, len(ds))))
    rows = []
    for i, e in enumerate(ds):
        options = list(e["choices"])
        gold_text = options[e["answer"]]
        choices_block, letter2text = format_choices(options, shuffle=shuffle, seed=i)
        gold_letter = next(l for l, t in letter2text.items() if t == gold_text)
        gt = make_ground_truth(
            answer=gold_letter, question_type="mc",
            answer_text=gold_text, choices=letter2text,
        )
        msgs = make_messages(context="", question=e["question"], choices_block=choices_block)
        # context empty -> "\n\nQuestion: ..."; strip leading blank lines for cleanliness
        msgs[1]["content"] = msgs[1]["content"].lstrip("\n")
        rows.append(make_sample(
            "mmlu", msgs, gt,
            question=e["question"], answer_text=gold_text, story="",
            extra_info={"subject": e.get("subject", "")},
        ))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/cleaned_tom/mmlu_test.parquet")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=2000,
                    help="cap rows (guardrail check doesn't need the full 14k)")
    ap.add_argument("--no_shuffle", action="store_true")
    args = ap.parse_args()
    rows = build(split=args.split, limit=args.limit, shuffle=not args.no_shuffle)
    write_parquet(rows, args.out)


if __name__ == "__main__":
    main()
