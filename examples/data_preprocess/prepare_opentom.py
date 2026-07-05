"""
Prepare the OpenToM benchmark (SeacowX/OpenToM) for BeRL evaluation.

OpenToM (OOD bucket): natural-language stories with first/second-order questions.
Question types (canonical opentom.json):
  - attitude          : 3-way {positive, negative, neutral}          -> MC
  - location-fo/so    : coarse "still in initial location?" Yes/No   -> binary
                        OR fine "where does X think it is?" (2 places)-> MC
  - multihop-fo/so    : fullness OR accessibility change (3 options) -> MC

HF: SeacowX/OpenToM  (opentom.json = list of 13708 dicts)

Output: data/cleaned_tom/opentom_test.parquet  (data_source = opentom_<type>)

Usage:
    python examples/data_preprocess/prepare_opentom.py \
        --out data/cleaned_tom/opentom_test.parquet
"""

import argparse

from huggingface_hub import hf_hub_download
import json

from tom_eval_common import (
    make_messages, format_choices, make_ground_truth, make_sample, write_parquet,
)

ATTITUDE = ["positive", "negative", "neutral"]
FULLNESS = ["more full", "less full", "equally full"]
ACCESS = ["more accessible", "less accessible", "equally accessible"]


def _norm(s: str) -> str:
    s = s.strip().lower()
    for pre in ("the ", "a ", "an "):
        if s.startswith(pre):
            s = s[len(pre):]
    return s


def _mc_row(data_source, context, question, options, gold_text, story, seed, extra):
    """Build one MC row; return None if gold not among options."""
    choices_block, letter2text = format_choices(options, shuffle=True, seed=seed)
    gold_letter = None
    for l, t in letter2text.items():
        if _norm(t) == _norm(gold_text):
            gold_letter = l
            gold_text = t
            break
    if gold_letter is None:
        return None
    gt = make_ground_truth(gold_letter, "mc", answer_text=gold_text, choices=letter2text)
    msgs = make_messages(context, question, choices_block)
    return make_sample(data_source, msgs, gt, question=question,
                       answer_text=gold_text, story=context, extra_info=extra)


def build(limit=None):
    path = hf_hub_download("SeacowX/OpenToM", "opentom.json", repo_type="dataset")
    data = json.load(open(path))
    if limit:
        data = data[:limit]

    rows, skipped = [], 0
    for i, e in enumerate(data):
        q = e["question"]
        qtype, qtext, ans = q["type"], q["question"], str(q["answer"])
        context = e.get("narrative") or e.get("plot", "")
        ds = f"opentom_{qtype.replace('-', '_')}"
        info = e.get("plot_info", {})
        extra = {"type": qtype, "mover": info.get("mover"), "eoi": info.get("eoi")}

        row = None
        if qtype == "attitude":
            row = _mc_row(ds, context, qtext, ATTITUDE, ans, context, i, extra)
        elif qtype.startswith("location"):
            if ans in ("Yes", "No"):
                gt = make_ground_truth(ans.lower(), "binary", answer_text=ans)
                msgs = make_messages(context, qtext)
                row = make_sample(ds, msgs, gt, question=qtext, answer_text=ans,
                                  story=context, extra_info=extra)
            else:
                places = [info.get("original_place", ""), info.get("move_to_place", "")]
                places = [p for p in places if p]
                row = _mc_row(ds, context, qtext, places, ans, context, i, extra)
        elif qtype.startswith("multihop"):
            opts = FULLNESS if "full" in qtext.lower() else ACCESS
            row = _mc_row(ds, context, qtext, opts, ans, context, i, extra)

        if row is None:
            skipped += 1
            continue
        rows.append(row)

    print(f"  (skipped {skipped} rows whose gold answer did not match candidate options)")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/cleaned_tom/opentom_test.parquet")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    rows = build(limit=args.limit)
    write_parquet(rows, args.out)


if __name__ == "__main__":
    main()
