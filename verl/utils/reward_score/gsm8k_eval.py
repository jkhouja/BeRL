"""
GSM8K *evaluation* scorer for BeRL (numeric-answer guardrail).

GSM8K is not a ToM task — it is a grade-school math word-problem benchmark used to
check that ToM / dialogue GRPO training does not degrade general numeric reasoning.

Unlike ``verl.utils.reward_score.gsm8k`` (the RL *training* scorer, which returns a
raw 0/1 and looks for the ``#### <n>`` GSM8K completion format), this scorer shares
the SAME interface and score range as the other BeRL eval scorers
(``explore_tom`` / ``tom_mc`` / ``fantom``):

    score = format_score (±format_reward) + answer_score (±answer_reward),  range [-3, +3]

so a fully-correct response scores exactly +3 and the ``ray_trainer`` validation
aggregation (which reports ``count(reward == 3) / total``) treats it identically to
the ToM benchmarks. It is parser-aware: the numeric answer is extracted from the
``<answer> </answer>`` region (or post-``</think>`` for tag-free models) rather than
from the raw ``#### n`` GSM8K string, matching how the model is prompted at eval.

``ground_truth`` may be a bare number string (e.g. ``"18"``) or a JSON dict
(as produced by ``examples/data_preprocess/tom_eval_common.make_ground_truth``):
    {"answer": "18", "question_type": "exact", ...}
"""

import json
import re
from typing import Any, Dict, Optional, Union

from verl.utils.reward_score.explore_tom import (
    extract_solution,
    validate_response_structure,
)
from verl.utils.reward_score.response_parser import ModelResponseParser


# Matches a signed integer or decimal, ignoring thousands separators / currency
# (those are stripped before matching).
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _extract_number(text: Optional[str]) -> Optional[str]:
    """Return the LAST numeric token in ``text`` (GSM8K's final answer convention).

    Strips thousands separators (``,``), currency (``$``) and percent signs so
    ``"$1,000"`` / ``"1,000"`` / ``"1000"`` all normalise to ``"1000"``. Returns
    ``None`` when no number is present.
    """
    if text is None:
        return None
    cleaned = text.replace(",", "").replace("$", "").replace("%", "")
    matches = _NUM_RE.findall(cleaned)
    if not matches:
        return None
    return matches[-1]


def _numbers_equal(a: Optional[str], b: Optional[str]) -> bool:
    """Numeric equality with a small float tolerance; string fallback."""
    if a is None or b is None:
        return False
    try:
        return abs(float(a) - float(b)) < 1e-4
    except (ValueError, TypeError):
        return str(a).strip() == str(b).strip()


def _parse_ground_truth(ground_truth: Union[Dict[str, Any], str]) -> str:
    """Extract the gold numeric answer from a raw string or the shared JSON dict."""
    gt_answer = ""
    if isinstance(ground_truth, dict):
        gt_answer = ground_truth.get("answer", ground_truth.get("expected_answer", ""))
    elif isinstance(ground_truth, str):
        gt = None
        try:
            gt = json.loads(ground_truth)
        except (json.JSONDecodeError, TypeError):
            gt = None
        gt_answer = gt.get("answer", "") if isinstance(gt, dict) else ground_truth
    else:
        gt_answer = ground_truth
    # gold may itself carry the GSM8K "#### n" tail or surrounding prose.
    return _extract_number(str(gt_answer))


def compute_score(solution_str: str,
                  ground_truth: Union[Dict[str, Any], str],
                  format_reward: int = 1,
                  answer_reward: float = 2.0,
                  require_answer_tags: bool = True,
                  parser: ModelResponseParser = None) -> float:
    print("\n" + "=" * 80)
    print(" Processing GSM8K-Eval Sample ".center(80, "="))

    gold_num = _parse_ground_truth(ground_truth)
    print(f"[Ground Truth] number={gold_num!r}")

    # ---- extract + format ----
    pred, processed_str = extract_solution(
        solution_str, require_answer_tags=require_answer_tags, parser=parser)
    print(f"\n[Model Response]\n{processed_str[:500]}")
    format_correct = validate_response_structure(
        processed_str, require_answer_tags=require_answer_tags, parser=parser)
    format_score = format_reward if format_correct else -abs(format_reward)

    # ---- answer ----
    answer_score = -answer_reward
    if format_correct and pred:
        pred_num = _extract_number(pred)
        is_correct = _numbers_equal(pred_num, gold_num)
        answer_score = answer_reward if is_correct else -answer_reward
        print(f"  Predicted number: {pred_num!r}  Correct: {is_correct}")
    else:
        print("  [Content Validation] Skipped (format error or empty answer)")

    total = format_score + answer_score
    print(f"  Format: {format_score}  Answer: {answer_score}  Total: {total}")
    print("=" * 80 + "\n")
    return total
