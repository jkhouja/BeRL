"""
Generic Theory-of-Mind multiple-choice / binary / exact answer scorer.

Same interface as explore_tom.compute_score / fantom.compute_score, used for the
OOD / Applied / Guardrail eval benchmarks that share the MC/binary answer format
(SimpleToM, ToMBench, BigToM, OpenToM, MMLU, ...).

``ground_truth`` is a JSON string (see examples/data_preprocess/tom_eval_common.py):
    {"answer": "<letter | yes/no | text>",
     "answer_text": "<gold option text>",
     "question_type": "mc" | "binary" | "list" | "exact",
     "wrong_answer": "...",
     "choices": {"A": "...", ...}}

Score = format_score (±1) + answer_score (±2), range [-3, +3] — matches the other
ToM scorers so eval aggregation is consistent.
"""

import json
from typing import Any, Dict, Union

from verl.utils.reward_score.explore_tom import (
    extract_solution,
    validate_response_structure,
    normalize_answer,
    check_answer_correctness,
)
from verl.utils.reward_score.fantom import _check_mc, _check_binary, _check_list
from verl.utils.reward_score.response_parser import ModelResponseParser


def _check_mc_or_text(predicted: str, correct_letter: str, answer_text: str,
                      choices: Dict[str, str]) -> bool:
    """MC match: accept the correct letter, '(X)', or the correct option TEXT.

    Guards against matching an option text that is merely a substring of a
    *distractor* by requiring that, when matching by text, the gold text matches
    at least as well as any other choice.
    """
    if correct_letter and _check_mc(predicted, correct_letter):
        return True

    norm_pred = normalize_answer(predicted)
    if answer_text:
        norm_gold = normalize_answer(answer_text)
        if norm_pred == norm_gold:
            return True
        # Substring match, but only if no *other* choice matches (avoid ambiguity)
        if norm_gold and norm_gold in norm_pred:
            for letter, txt in (choices or {}).items():
                if letter.strip().upper() == str(correct_letter).strip().upper():
                    continue
                norm_other = normalize_answer(txt)
                if norm_other and norm_other in norm_pred and norm_other != norm_gold:
                    return False  # ambiguous — a distractor also appears
            return True
    return False


def compute_score(solution_str: str,
                  ground_truth: Union[Dict[str, Any], str],
                  format_reward: int = 1,
                  answer_reward: float = 2.0,
                  require_answer_tags: bool = True,
                  parser: ModelResponseParser = None) -> float:
    print("\n" + "=" * 80)
    print(" Processing ToM-MC Sample ".center(80, "="))

    # ---- parse ground truth ----
    gt_answer, answer_text, question_type, wrong_answer, choices = "", "", "mc", "", {}
    if isinstance(ground_truth, str):
        try:
            gt = json.loads(ground_truth)
        except (json.JSONDecodeError, TypeError):
            gt = None
        if isinstance(gt, dict):
            gt_answer = gt.get("answer", "")
            answer_text = gt.get("answer_text", "")
            question_type = gt.get("question_type", "mc")
            wrong_answer = gt.get("wrong_answer", "")
            choices = gt.get("choices", {}) or {}
        else:
            gt_answer = ground_truth
    elif isinstance(ground_truth, dict):
        gt_answer = ground_truth.get("answer", ground_truth.get("expected_answer", ""))
        answer_text = ground_truth.get("answer_text", "")
        question_type = ground_truth.get("question_type", "mc")
        wrong_answer = ground_truth.get("wrong_answer", "")
        choices = ground_truth.get("choices", {}) or {}
    else:
        gt_answer = ground_truth

    print(f"[Ground Truth] answer={gt_answer!r} text={answer_text!r} type={question_type}")

    # ---- extract + format ----
    pred, processed_str = extract_solution(solution_str, require_answer_tags=require_answer_tags, parser=parser)
    print(f"\n[Model Response]\n{processed_str[:500]}")
    format_correct = validate_response_structure(processed_str, require_answer_tags=require_answer_tags, parser=parser)
    format_score = format_reward if format_correct else -abs(format_reward)

    # ---- answer ----
    answer_score = -answer_reward
    if format_correct and pred:
        if question_type == "binary":
            is_correct = _check_binary(pred, gt_answer)
        elif question_type == "list":
            is_correct = _check_list(pred, gt_answer, wrong_answer)
        elif question_type == "exact":
            # Use explore_tom's matcher so open-ended answers (e.g. ExploreToM-Infilled)
            # are scored identically to the in-distribution explore_tom eval:
            # exact normalized match OR prediction ends with the gold answer.
            is_correct, _ = check_answer_correctness(pred, gt_answer)
        else:  # mc (default)
            is_correct = _check_mc_or_text(pred, gt_answer, answer_text, choices)
        answer_score = answer_reward if is_correct else -answer_reward
        print(f"  Answer correct: {is_correct}")
    else:
        print("  [Content Validation] Skipped (format error or empty answer)")

    total = format_score + answer_score
    print(f"  Format: {format_score}  Answer: {answer_score}  Total: {total}")
    print("=" * 80 + "\n")
    return total
