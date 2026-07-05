"""
Prepare the ExploreToM-Infilled robustness eval (facebook/ExploreToM).

The in-distribution explore_tom eval uses the *symbolic* story_structure. This
robustness variant uses the natural-language ``infilled_story`` for the same
question/answer, testing whether ToM reasoning survives narrative surface form.

Open-ended QA (no answer choices), matching the existing explore_tom eval:
  - yes/no answers      -> question_type "binary"
  - everything else     -> question_type "exact"  (normalized text match)

HF: facebook/ExploreToM  (train split, 13309 rows). We sample a test-sized subset.

Output: data/cleaned_tom/exploretom_infilled_test.parquet
        (data_source = exploretom_infilled)

Usage:
    python examples/data_preprocess/prepare_exploretom_infilled.py \
        --out data/cleaned_tom/exploretom_infilled_test.parquet --limit 1500
"""

import argparse

import datasets

from tom_eval_common import make_messages, make_ground_truth, make_sample, write_parquet


def build(limit=1500, seed=42):
    ds = datasets.load_dataset("facebook/ExploreToM", split="train")
    ds = ds.shuffle(seed=seed)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    rows = []
    for e in ds:
        story = e["infilled_story"]
        question = e["question"]
        ans = str(e["expected_answer"]).strip()
        if not ans:
            continue
        if ans.lower() in ("yes", "no"):
            qtype, gold = "binary", ans.lower()
        else:
            qtype, gold = "exact", ans
        gt = make_ground_truth(gold, qtype, answer_text=ans)
        msgs = make_messages(story, question)  # open-ended, no choices
        rows.append(make_sample(
            "exploretom_infilled", msgs, gt,
            question=question, answer_text=ans, story=story,
            extra_info={"nth_order": e.get("qprop=nth_order"),
                        "story_type": e.get("param=story_type")},
        ))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/cleaned_tom/exploretom_infilled_test.parquet")
    ap.add_argument("--limit", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rows = build(limit=args.limit, seed=args.seed)
    write_parquet(rows, args.out)


if __name__ == "__main__":
    main()
