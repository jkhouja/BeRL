"""
Prepare the SimpleToM benchmark (allenai/SimpleToM) for BeRL evaluation.

SimpleToM (Applied bucket): short story + a multiple-choice question across three
subsets — mental-state, behavior, and judgment QA — testing whether a model can
apply implicit mental-state inferences.

HF: allenai/SimpleToM  (files: {mental-state,behavior,judgment}-qa/test.jsonl)
Each row: id, story, question, scenario_name, choices (str repr of
          {'text': [...], 'label': ['A','B']}), answerKey ('A'/'B').

Output: data/cleaned_tom/simpletom_test.parquet  (data_source = simpletom_<subset>)

Usage:
    python examples/data_preprocess/prepare_simpletom.py \
        --out data/cleaned_tom/simpletom_test.parquet
"""

import argparse
import ast
import json
import os

from huggingface_hub import hf_hub_download

from tom_eval_common import make_messages, make_ground_truth, make_sample, write_parquet

SUBSETS = {
    "mental-state-qa/test.jsonl": "simpletom_mental",
    "behavior-qa/test.jsonl": "simpletom_behavior",
    "judgment-qa/test.jsonl": "simpletom_judgment",
}


def _parse_choices(raw):
    """choices is stored as a python-dict string: {'text': [...], 'label': ['A','B']}."""
    if isinstance(raw, dict):
        return raw
    return ast.literal_eval(raw)


def build(add_hint=False, wo_think=False, limit=None):
    rows = []
    for fn, data_source in SUBSETS.items():
        path = hf_hub_download("allenai/SimpleToM", fn, repo_type="dataset")
        with open(path) as f:
            entries = [json.loads(l) for l in f if l.strip()]
        if limit:
            entries = entries[:limit]
        for e in entries:
            ch = _parse_choices(e["choices"])
            texts, labels = ch["text"], ch["label"]
            letter2text = {lab: txt for lab, txt in zip(labels, texts)}
            choices_block = "\n".join(f"({lab}) {txt}" for lab, txt in zip(labels, texts))
            answer_letter = e["answerKey"]
            answer_text = letter2text.get(answer_letter, "")
            gt = make_ground_truth(
                answer=answer_letter, question_type="mc",
                answer_text=answer_text, choices=letter2text,
            )
            msgs = make_messages(e["story"], e["question"], choices_block,
                                 add_hint=add_hint, wo_think=wo_think)
            rows.append(make_sample(
                data_source, msgs, gt,
                question=e["question"], answer_text=answer_text, story=e["story"],
                extra_info={"id": e.get("id", ""), "scenario": e.get("scenario_name", ""),
                            "subset": data_source},
            ))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/cleaned_tom/simpletom_test.parquet")
    ap.add_argument("--add_hint", action="store_true")
    ap.add_argument("--wo_think", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="cap rows per subset (debug)")
    args = ap.parse_args()

    rows = build(add_hint=args.add_hint, wo_think=args.wo_think, limit=args.limit)
    write_parquet(rows, args.out)


if __name__ == "__main__":
    main()
