"""
Unit tests for the generic ToM MC/binary/exact eval scorer (verl.utils.reward_score.tom_mc).

Verifies correct answers score positively and wrong/other answers score negatively,
across the answer formats models actually emit: bare letter, "(B)", option text,
and yes/no. Offline (no network).

Run: python tests/reward_score/test_tom_mc.py   (or: pytest tests/reward_score/test_tom_mc.py)
"""

import json
import sys

from verl.utils.reward_score import tom_mc


def _resp(answer_body: str) -> str:
    """Simulate a model response with the eval prompt's <think>/<answer> format."""
    return f"Assistant: <think>some reasoning here</think><answer>{answer_body}</answer>"


def _gt(**kw) -> str:
    base = {"answer": "", "answer_text": "", "question_type": "mc", "wrong_answer": "", "choices": {}}
    base.update(kw)
    return json.dumps(base)


CASES = [
    # name, ground_truth, answer_body, expect_positive
    ("mc_letter_correct",   _gt(answer="B", answer_text="No", choices={"A": "Yes", "B": "No"}), "B", True),
    ("mc_paren_correct",    _gt(answer="B", answer_text="No", choices={"A": "Yes", "B": "No"}), "(B)", True),
    ("mc_text_correct",     _gt(answer="B", answer_text="No", choices={"A": "Yes", "B": "No"}), "No", True),
    ("mc_letter_wrong",     _gt(answer="B", answer_text="No", choices={"A": "Yes", "B": "No"}), "A", False),
    ("mc_text_wrong",       _gt(answer="B", answer_text="No", choices={"A": "Yes", "B": "No"}), "Yes", False),
    ("mc_text_4opt_correct", _gt(answer="C", answer_text="the green box",
                                 choices={"A": "the red box", "B": "the blue box",
                                          "C": "the green box", "D": "the yellow box"}),
     "the green box", True),
    ("binary_yes_correct",  _gt(answer="yes", question_type="binary"), "Yes", True),
    ("binary_no_correct",   _gt(answer="no", question_type="binary"), "No.", True),
    ("binary_wrong",        _gt(answer="yes", question_type="binary"), "no", False),
    ("exact_correct",       _gt(answer="the basket", question_type="exact"), "the basket", True),
    ("exact_wrong",         _gt(answer="the basket", question_type="exact"), "the box", False),
]


def run():
    failures = []
    for name, gt, body, expect_pos in CASES:
        score = tom_mc.compute_score(_resp(body), gt, require_answer_tags=True, parser=None)
        # format is correct in all _resp(...) strings, so answer sign decides:
        # correct  -> +1 + 2 = 3 ; wrong -> +1 - 2 = -1
        ok = (score > 0) == expect_pos
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name}: score={score} expect_positive={expect_pos}")
        if not ok:
            failures.append(name)

    # A format-broken response (no tags) must never score positive.
    bad = tom_mc.compute_score("Assistant: the answer is B", _gt(answer="B", answer_text="No"),
                               require_answer_tags=True, parser=None)
    if bad > 0:
        failures.append("format_broken_should_be_nonpositive")
    print(f"[{'PASS' if bad <= 0 else 'FAIL'}] format_broken: score={bad}")

    print(f"\n{len(CASES) + 1 - len(failures)}/{len(CASES) + 1} checks passed")
    if failures:
        print("FAILURES:", failures)
        return 1
    return 0


def test_tom_mc_scoring():
    assert run() == 0


if __name__ == "__main__":
    sys.exit(run())
