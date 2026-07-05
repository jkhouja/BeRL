"""
Prepare the ToMBench benchmark (ycfNTU/tombench_merged) for BeRL evaluation.

ToMBench (OOD bucket): English-translated multiple-choice ToM questions spanning
many abilities (false belief, faux pas, intention, etc.). 2- to 4-option MC.

HF: ycfNTU/tombench_merged  (merged_dataset.csv)
Relevant columns: STORY, QUESTION, OPTION-A..D, 'ANSWER' (letter), ABILITY.
(The csv also carries a prior run's model_output/score columns, which are ignored.)

Output: data/cleaned_tom/tombench_test.parquet  (data_source = tombench)

Usage:
    python examples/data_preprocess/prepare_tombench.py \
        --out data/cleaned_tom/tombench_test.parquet
"""

import argparse

from huggingface_hub import hf_hub_download
import pandas as pd

from tom_eval_common import make_messages, make_ground_truth, make_sample, write_parquet

ANSWER_COL = "答案\nANSWER"
ABILITY_COL = "能力\nABILITY"
OPTION_COLS = [("A", "OPTION-A"), ("B", "OPTION-B"), ("C", "OPTION-C"), ("D", "OPTION-D")]


def build(limit=None):
    path = hf_hub_download("ycfNTU/tombench_merged", "merged_dataset.csv", repo_type="dataset")
    df = pd.read_csv(path)
    if limit:
        df = df.head(limit)

    rows, skipped = [], 0
    for _, e in df.iterrows():
        story = str(e.get("STORY", "") or "")
        question = str(e.get("QUESTION", "") or "")
        gold_letter = str(e.get(ANSWER_COL, "") or "").strip().upper()
        if not question or gold_letter not in ("A", "B", "C", "D"):
            skipped += 1
            continue

        letter2text, lines = {}, []
        for letter, col in OPTION_COLS:
            val = e.get(col)
            if pd.isna(val) or str(val).strip() == "":
                continue
            txt = str(val).strip()
            letter2text[letter] = txt
            lines.append(f"({letter}) {txt}")

        if gold_letter not in letter2text:
            skipped += 1
            continue

        gt = make_ground_truth(
            gold_letter, "mc",
            answer_text=letter2text[gold_letter], choices=letter2text,
        )
        msgs = make_messages(story, question, "\n".join(lines))
        rows.append(make_sample(
            "tombench", msgs, gt,
            question=question, answer_text=letter2text[gold_letter], story=story,
            extra_info={"ability": str(e.get(ABILITY_COL, "") or ""),
                        "sheet": str(e.get("sheet_name", "") or "")},
        ))

    print(f"  (skipped {skipped} rows with missing question/options/gold)")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/cleaned_tom/tombench_test.parquet")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    rows = build(limit=args.limit)
    write_parquet(rows, args.out)


if __name__ == "__main__":
    main()
