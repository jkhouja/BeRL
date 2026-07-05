"""
Prepare the Ullman (2023) perturbation set for BeRL robustness evaluation.

These are the canonical "small variations that break brittle ToM" false-belief
perturbations (transparent bag, cannot read, trusted-friend override, etc.).
Each is a 3-way MC; a robust ToM reasoner should answer all correctly while
pattern-matchers fail on the perturbed variants.

Source: GitHub cicl-stanford/procedural-evals-tom  ->  data/expert_data/ullman.csv
(semicolon-delimited, no header: story ; question ; correct ; distractor1 ; distractor2)

Output: data/cleaned_tom/ullman_perturbed_test.parquet  (data_source = ullman_perturbed)

Usage:
    python examples/data_preprocess/prepare_ullman_perturbed.py \
        --out data/cleaned_tom/ullman_perturbed_test.parquet
"""

import argparse
import csv
import io
import urllib.request

from tom_eval_common import format_choices, make_messages, make_ground_truth, make_sample, write_parquet

CSV_URL = ("https://raw.githubusercontent.com/cicl-stanford/"
           "procedural-evals-tom/main/data/expert_data/ullman.csv")


def build():
    with urllib.request.urlopen(CSV_URL, timeout=60) as resp:
        text = resp.read().decode("utf-8")
    reader = csv.reader(io.StringIO(text), delimiter=";")

    rows = []
    for i, rec in enumerate(reader):
        rec = [c.strip() for c in rec if c.strip() != ""]
        if len(rec) < 4:
            continue
        story, question = rec[0], rec[1]
        options = rec[2:]           # rec[2] is the correct answer
        gold_text = options[0]
        choices_block, letter2text = format_choices(options, shuffle=True, seed=i)
        gold_letter = next(l for l, t in letter2text.items() if t == gold_text)
        gt = make_ground_truth(gold_letter, "mc", answer_text=gold_text, choices=letter2text)
        msgs = make_messages(story, question, choices_block)
        rows.append(make_sample(
            "ullman_perturbed", msgs, gt,
            question=question, answer_text=gold_text, story=story,
            extra_info={"idx": i},
        ))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/cleaned_tom/ullman_perturbed_test.parquet")
    args = ap.parse_args()
    rows = build()
    write_parquet(rows, args.out)


if __name__ == "__main__":
    main()
