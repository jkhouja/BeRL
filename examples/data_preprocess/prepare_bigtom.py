"""
Prepare the BigToM benchmark (Gandhi et al. 2023) for BeRL evaluation.

BigToM (OOD bucket): procedurally-generated causal ToM stories. Each raw record is
assembled into condition-specific 2-choice questions. We reconstruct the canonical
conditions (init_belief hidden) exactly as the official generate_conditions.py does:

  variables : forward_belief, forward_action, backward_belief
  conditions: true_belief (agent aware), false_belief (agent unaware)

Source: GitHub cicl-stanford/procedural-evals-tom  ->  data/bigtom/bigtom.csv
(semicolon-delimited, no header; 17 fields). No HF data mirror exists.

Output: data/cleaned_tom/bigtom_test.parquet  (data_source = bigtom_<variable>)

Usage:
    python examples/data_preprocess/prepare_bigtom.py \
        --out data/cleaned_tom/bigtom_test.parquet
"""

import argparse
import csv
import io
import urllib.request

from tom_eval_common import format_choices, make_messages, make_ground_truth, make_sample, write_parquet

CSV_URL = ("https://raw.githubusercontent.com/cicl-stanford/"
           "procedural-evals-tom/main/data/bigtom/bigtom.csv")

# Column order from the official generate_conditions.py `list_var`.
FIELDS = ["Story", "Aware of event", "Not Aware of event", "Action aware", "Action not aware",
          "Belief Question", "Desire Question", "Action Question",
          "Belief Answer Aware", "Desire Answer Aware", "Action Answer Aware",
          "Belief Answer not Aware", "Desire Answer not Aware", "Action Answer not Aware",
          "Random Event", "Aware of random event", "Not aware of random event"]


def _story_hidden_belief(d):
    """init_belief=0: drop the explicit initial-belief sentence (part 3)."""
    parts = d["Story"].split(".")
    return parts[0] + "." + parts[1] + "." + parts[2] + "." + parts[4] + "."


def _rows_for_record(d, idx):
    """Yield (data_source, story, question, true_answer, wrong_answer) tuples."""
    story = _story_hidden_belief(d)
    specs = [
        ("bigtom_forward_belief",  d["Belief Question"],
         d["Belief Answer Aware"], d["Belief Answer not Aware"],
         d["Aware of event"], d["Not Aware of event"]),
        ("bigtom_forward_action",  d["Action Question"],
         d["Action Answer Aware"], d["Action Answer not Aware"],
         d["Aware of event"], d["Not Aware of event"]),
        ("bigtom_backward_belief", d["Belief Question"],
         d["Belief Answer Aware"], d["Belief Answer not Aware"],
         d["Action aware"], d["Action not aware"]),
    ]
    for ds, question, ans_aware, ans_unaware, suffix_true, suffix_false in specs:
        # true_belief: agent aware -> gold is the "aware" answer
        yield (ds, "true_belief", f"{story} {suffix_true}", question, ans_aware, ans_unaware)
        # false_belief: agent unaware -> gold is the "not aware" answer
        yield (ds, "false_belief", f"{story} {suffix_false}", question, ans_unaware, ans_aware)


def build(limit=None):
    with urllib.request.urlopen(CSV_URL, timeout=60) as resp:
        text = resp.read().decode("utf-8")
    reader = csv.reader(io.StringIO(text), delimiter=";")
    records = list(reader)
    if limit:
        records = records[:limit]

    rows, skipped, seed = [], 0, 0
    for ridx, rec in enumerate(records):
        if len(rec) < len(FIELDS):
            skipped += 1
            continue
        d = {FIELDS[i]: rec[i] for i in range(len(FIELDS))}
        try:
            gen = list(_rows_for_record(d, ridx))
        except IndexError:
            skipped += 1
            continue
        for ds, cond, story, question, true_ans, wrong_ans in gen:
            true_ans, wrong_ans = true_ans.strip(), wrong_ans.strip()
            if not true_ans or true_ans == wrong_ans:
                continue
            choices_block, letter2text = format_choices([true_ans, wrong_ans],
                                                         shuffle=True, seed=seed)
            seed += 1
            gold_letter = next(l for l, t in letter2text.items() if t == true_ans)
            gt = make_ground_truth(gold_letter, "mc",
                                   answer_text=true_ans, choices=letter2text)
            msgs = make_messages(story.strip(), question, choices_block)
            rows.append(make_sample(
                ds, msgs, gt, question=question, answer_text=true_ans, story=story.strip(),
                extra_info={"condition": cond, "variable": ds},
            ))
    print(f"  (skipped {skipped} malformed records)")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/cleaned_tom/bigtom_test.parquet")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    rows = build(limit=args.limit)
    write_parquet(rows, args.out)


if __name__ == "__main__":
    main()
