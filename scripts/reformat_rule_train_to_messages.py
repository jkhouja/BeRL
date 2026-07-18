"""Reformat the legacy rule-based ToM training parquet into the native message
format used by the eval sets, fixing the train/eval format mismatch.

Background
----------
The legacy ``ToM_train_HiEx_hint.parquet`` stores each ``prompt`` as a SINGLE
``user`` message whose content is a fully-rendered Qwen ChatML string ending in
``<|im_start|>assistant\\n<think>`` (a baked ``<think>`` prefill). When this is
fed through ``RLHFDataset`` with the default ``prompt_is_text=False``, the
tokenizer re-applies ``apply_chat_template(add_generation_prompt=True)`` and
DOUBLE-WRAPS the text: the rendered prompt then contains TWO
``<|im_start|>assistant`` markers, the ``<think>`` is stranded inside a nested
assistant turn, and the real generation point has NO prefill. Meanwhile the eval
parquets (``ToM_test_HiExTi_hint_v3.parquet``) use the clean ``[system, user]``
message format with no prefill. This train/eval mismatch is the root cause of the
apparent ToM "regression" (the model drifts off the ``<think>``/``<answer>``
format the strict scorer's format gate requires at eval time).

Fix
---
Re-derive ``prompt`` from the row's ``story`` + ``question`` using the SAME
builder that produced the eval v3 parquet (``merge_tom.make_messages`` with
``add_hint=True, concise_answer=True``), so the training prompt is byte-for-byte
structurally identical to the eval prompt (same ``cot_eval`` system prompt +
room-witness hint + "output ONLY the key noun" note, same
``Read the following story ...`` user template, no ``<think>`` prefill). All
other columns (``answer``, ``data_source``, ``reward_model`` with plain-string
``ground_truth``, ``question``, ``story``, ``ability``, ``extra_info``) are
preserved unchanged so the rule scorer keeps working.

Usage
-----
    python scripts/reformat_rule_train_to_messages.py \
        --in  data/cleaned_tom/ToM_train_HiEx_hint.parquet \
        --out data/cleaned_tom/ToM_train_HiEx_hint_v3.parquet
"""

import argparse
import os
import sys

import pandas as pd

# Reuse the exact eval-prompt builder (no hardcoded prompt strings here).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples", "data_preprocess"))
from merge_tom import make_messages  # noqa: E402


def reformat(in_path: str, out_path: str, add_hint: bool = True,
             concise_answer: bool = True, wo_think: bool = False) -> None:
    df = pd.read_parquet(in_path)
    required = {"story", "question", "answer", "data_source", "reward_model"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"input parquet missing required columns: {sorted(missing)}")

    def build_prompt(row):
        return make_messages(
            story=row["story"],
            question=row["question"],
            add_hint=add_hint,
            wo_think=wo_think,
            concise_answer=concise_answer,
        )

    df = df.copy()
    df["prompt"] = df.apply(build_prompt, axis=1)
    if "ability" not in df.columns:
        df["ability"] = "theory_of_mind"
    if "extra_info" not in df.columns:
        df["extra_info"] = [{"key": "dummy"} for _ in range(len(df))]

    cols = ["data_source", "prompt", "ability", "reward_model",
            "extra_info", "answer", "question", "story"]
    df = df[[c for c in cols if c in df.columns]]

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    df.to_parquet(out_path)
    print(f"wrote {len(df)} rows -> {out_path}")
    print("by data_source:", df["data_source"].value_counts().to_dict())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path",
                    default="data/cleaned_tom/ToM_train_HiEx_hint.parquet")
    ap.add_argument("--out", dest="out_path",
                    default="data/cleaned_tom/ToM_train_HiEx_hint_v3.parquet")
    ap.add_argument("--no_hint", action="store_true",
                    help="Omit the room-witness hint (default: include, matches eval v3)")
    ap.add_argument("--no_concise", action="store_true",
                    help="Omit the concise-answer note (default: include, matches eval v3)")
    ap.add_argument("--wo_think", action="store_true",
                    help="Use the tag-free system prompt variant")
    args = ap.parse_args()
    reformat(args.in_path, args.out_path,
             add_hint=not args.no_hint,
             concise_answer=not args.no_concise,
             wo_think=args.wo_think)
