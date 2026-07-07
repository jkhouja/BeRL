"""
Unit tests for the GSM8K numeric-guardrail eval scorer
(verl.utils.reward_score.gsm8k_eval).

Verifies:
  * correct numbers score +3, wrong numbers score -1 (format ok, answer wrong),
  * robust numeric normalisation ($, commas, %, decimals, negatives, units),
  * strict boundaries (e.g. "180" must NOT match gold "18"),
  * both bare-string and JSON ground_truth,
  * format-broken responses never score positive,
  * tag-free models (Qwen3/Gemma-style, REQUIRE_ANSWER_TAGS=False) via a parser.

Offline (no network). Run:
  python tests/reward_score/test_gsm8k_eval.py
  (or: pytest tests/reward_score/test_gsm8k_eval.py)
"""

import json
import sys

from verl.utils.reward_score import gsm8k_eval
from verl.utils.reward_score.response_parser import get_parser


def _resp(answer_body: str) -> str:
    """Qwen2.5-style response with the eval prompt's <think>/<answer> format."""
    return f"Assistant: <think>some reasoning here</think><answer>{answer_body}</answer>"


def _gt_json(ans: str) -> str:
    return json.dumps({
        "answer": ans, "answer_text": ans, "question_type": "exact",
        "wrong_answer": "", "choices": {},
    })


# name, ground_truth, answer_body, expect_positive
CASES = [
    ("int_correct",            "18",          "18",            True),
    ("int_wrong",              "18",          "17",            False),
    ("units_suffix_correct",   "18",          "18 apples",     True),
    ("prose_prefix_correct",   "18",          "The answer is 18", True),
    ("currency_commas",        "1000",        "$1,000",        True),
    ("percent",                "50",          "50%",           True),
    ("decimal_equiv",          "18",          "18.0",          True),
    ("negative_correct",       "-5",          "-5",            True),
    ("superstring_wrong",      "18",          "180",           False),   # must NOT match
    ("substring_wrong",        "180",         "18",            False),
    ("gold_with_hash_tail",    "#### 42",     "42",            True),    # gold carries GSM8K tail
    ("json_gt_correct",        _gt_json("400"), "400",         True),
    ("json_gt_wrong",          _gt_json("400"), "399",         False),
]


def run():
    failures = []
    for name, gt, body, expect_pos in CASES:
        score = gsm8k_eval.compute_score(_resp(body), gt, require_answer_tags=True, parser=None)
        # format is always correct in _resp(...), so answer sign decides:
        # correct -> +1 + 2 = 3 ; wrong -> +1 - 2 = -1
        ok = (score > 0) == expect_pos
        if expect_pos:
            ok = ok and (score == 3)
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name}: score={score} expect_positive={expect_pos}")
        if not ok:
            failures.append(name)

    # Format-broken (no tags) must never score positive.
    bad = gsm8k_eval.compute_score("Assistant: the answer is 18", "18",
                                   require_answer_tags=True, parser=None)
    print(f"[{'PASS' if bad <= 0 else 'FAIL'}] format_broken: score={bad}")
    if bad > 0:
        failures.append("format_broken_should_be_nonpositive")

    # Tag-free model (Qwen3): <think>...</think> then bare number, no <answer> tags.
    parser = get_parser("qwen3")
    qwen3_ok = f"<|im_start|>assistant\n<think>work it out</think>18"
    s_ok = gsm8k_eval.compute_score(qwen3_ok, "18", require_answer_tags=False, parser=parser)
    print(f"[{'PASS' if s_ok == 3 else 'FAIL'}] qwen3_tagfree_correct: score={s_ok}")
    if s_ok != 3:
        failures.append("qwen3_tagfree_correct")

    qwen3_wrong = f"<|im_start|>assistant\n<think>work it out</think>17"
    s_wrong = gsm8k_eval.compute_score(qwen3_wrong, "18", require_answer_tags=False, parser=parser)
    print(f"[{'PASS' if s_wrong <= 0 else 'FAIL'}] qwen3_tagfree_wrong: score={s_wrong}")
    if s_wrong > 0:
        failures.append("qwen3_tagfree_wrong")

    total = len(CASES) + 3
    print(f"\n{total - len(failures)}/{total} checks passed")
    if failures:
        print("FAILURES:", failures)
        return 1
    return 0


def test_gsm8k_eval_scoring():
    assert run() == 0


if __name__ == "__main__":
    sys.exit(run())
