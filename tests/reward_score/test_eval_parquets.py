"""
Integration test: every built ToM eval parquet must score its own gold answer
positively via the routed scorer, and score an obviously-wrong answer non-positively.

Parquets live under data/cleaned_tom/ (git-ignored, produced by the prepare_*.py
scripts). Each present file is checked; missing files are skipped (not failed) so
the test is usable in environments where the data hasn't been generated.

Run: python tests/reward_score/test_eval_parquets.py
"""

import json
import os
import sys

import pandas as pd

from verl.trainer.main_ppo import _select_rm_score_fn

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "cleaned_tom")

PARQUETS = [
    "simpletom_test.parquet",
    "tombench_test.parquet",
    "opentom_test.parquet",
    "bigtom_test.parquet",
    "dyntom_test.parquet",
    "mmlu_test.parquet",
    "ullman_perturbed_test.parquet",
    "exploretom_infilled_test.parquet",
]

N_SAMPLE = 100


def _resp(body):
    return f"Assistant: <think>reasoning</think><answer>{body}</answer>"


def _wrong_body(gt_json):
    """Pick an answer guaranteed different from gold."""
    gt = json.loads(gt_json)
    choices = gt.get("choices") or {}
    for letter, text in choices.items():
        if letter != gt["answer"] and text != gt.get("answer_text"):
            return letter
    # binary / exact fallbacks
    if gt.get("question_type") == "binary":
        return "no" if gt["answer"].lower() == "yes" else "yes"
    return "definitely-not-the-answer-xyz"


def run():
    failures, checked = [], 0
    for fn in PARQUETS:
        path = os.path.join(DATA_DIR, fn)
        if not os.path.exists(path):
            print(f"[SKIP] {fn} (not generated)")
            continue
        df = pd.read_parquet(path)
        sample = df.sample(min(N_SAMPLE, len(df)), random_state=0)
        n_pos = n_neg_ok = 0
        for _, r in sample.iterrows():
            ds = r["data_source"]
            gt = r["reward_model"]["ground_truth"]
            score_fn, _ = _select_rm_score_fn(ds)
            s_gold = score_fn(_resp(r["answer"]), gt, require_answer_tags=True, parser=None)
            s_wrong = score_fn(_resp(_wrong_body(gt)), gt, require_answer_tags=True, parser=None)
            n_pos += s_gold > 0
            n_neg_ok += s_wrong <= 0
        gold_ok = n_pos == len(sample)
        wrong_ok = n_neg_ok == len(sample)
        status = "PASS" if (gold_ok and wrong_ok) else "FAIL"
        print(f"[{status}] {fn}: gold+ {n_pos}/{len(sample)}  wrong<=0 {n_neg_ok}/{len(sample)}")
        checked += 1
        if not (gold_ok and wrong_ok):
            failures.append(fn)

    if checked == 0:
        print("No parquets present; nothing to check.")
    if failures:
        print("FAILURES:", failures)
        return 1
    return 0


def test_eval_parquets_score_gold_positive():
    assert run() == 0


if __name__ == "__main__":
    sys.exit(run())
