"""
FANToM scoring — same interface as explore_tom.compute_score.

Score = format_score (±1) + answer_score (±2), range [-3, +3].

Question types:
  - belief_mc: check extracted answer matches correct letter (A/B)
  - binary (*_binary): normalize to yes/no, compare
  - list (*_list): check all correct names present, no wrong names
"""

import re
import json
from typing import Union, Dict, Any

from verl.utils.reward_score.explore_tom import (
    extract_solution,
    validate_response_structure,
    normalize_answer,
)


def _check_mc(predicted: str, correct_letter: str) -> bool:
    """Check if prediction matches the correct MC letter."""
    pred = predicted.strip().upper()
    letter = correct_letter.strip().upper()
    # Accept if starts with the letter, or contains "(X)" pattern
    if pred.startswith(letter):
        return True
    if f"({letter})" in pred:
        return True
    # Also accept if the entire normalized answer is just the letter
    if normalize_answer(predicted).strip().upper() == letter:
        return True
    return False


def _check_binary(predicted: str, correct: str) -> bool:
    """Check yes/no answer."""
    norm_pred = normalize_answer(predicted)
    norm_truth = normalize_answer(correct)
    # Map to canonical yes/no
    yes_set = {"yes", "y", "true"}
    no_set = {"no", "n", "false"}
    pred_val = "yes" if norm_pred in yes_set else ("no" if norm_pred in no_set else norm_pred)
    truth_val = "yes" if norm_truth in yes_set else ("no" if norm_truth in no_set else norm_truth)
    return pred_val == truth_val


def _check_list(predicted: str, correct: str, wrong: str) -> bool:
    """Check list answer: all correct names present, no wrong names."""
    norm_pred = normalize_answer(predicted)

    # Parse correct names
    correct_names = [normalize_answer(n) for n in re.split(r'[,;]', correct) if n.strip()]
    # Parse wrong names
    wrong_names = [normalize_answer(n) for n in re.split(r'[,;]', wrong) if n.strip()] if wrong else []

    # Handle "none" case
    if normalize_answer(correct) == "none":
        # Prediction should also be none/empty and not contain any wrong names
        if norm_pred in ("none", "no one", "nobody", ""):
            return True
        return False

    # All correct names must appear
    for name in correct_names:
        if name not in norm_pred:
            return False

    # No wrong names should appear
    for name in wrong_names:
        if name in norm_pred:
            return False

    return True


def compute_score(solution_str: str,
                  ground_truth: Union[Dict[str, Any], str],
                  format_reward: int = 1,
                  answer_reward: float = 2.0,
                  require_answer_tags: bool = True) -> float:
    """Compute score for a FANToM sample.

    Args:
        solution_str: Raw model response string
        ground_truth: The correct answer (string) or dict with ground_truth fields
        format_reward: Points for format correctness (±1)
        answer_reward: Points for answer correctness (±2)

    Returns:
        Total score in [-3, +3]
    """
    print("\n" + "=" * 80)
    print(" Processing FANToM Sample ".center(80, "="))

    # Extract ground truth answer — ground_truth may be a JSON string, dict, or plain string
    if isinstance(ground_truth, str):
        try:
            gt_parsed = json.loads(ground_truth)
            if isinstance(gt_parsed, dict):
                gt_answer = gt_parsed.get("answer", "")
                question_type = gt_parsed.get("question_type", "")
                wrong_answer = gt_parsed.get("wrong_answer", "")
            else:
                gt_answer = ground_truth
                question_type = ""
                wrong_answer = ""
        except (json.JSONDecodeError, TypeError):
            gt_answer = ground_truth
            question_type = ""
            wrong_answer = ""
    elif isinstance(ground_truth, dict):
        gt_answer = ground_truth.get("answer", ground_truth.get("ground_truth", ground_truth.get("expected_answer", "")))
        question_type = ground_truth.get("question_type", "")
        wrong_answer = ground_truth.get("wrong_answer", "")
    else:
        gt_answer = ground_truth
        question_type = ""
        wrong_answer = ""

    print(f"[Ground Truth] {gt_answer}  (type={question_type})")

    # Extract model answer
    answer_text, processed_str = extract_solution(solution_str, require_answer_tags=require_answer_tags)
    print(f"\n[Model Response]\n{processed_str[:500]}")

    # Format validation
    format_correct = validate_response_structure(processed_str, require_answer_tags=require_answer_tags)
    format_score = format_reward if format_correct else -abs(format_reward)
    print(f"\n  Format score: {format_score}")

    # Answer validation
    answer_score = -answer_reward  # default: wrong
    if format_correct and answer_text:
        is_correct = False
        if "mc" in question_type or "belief" in question_type:
            is_correct = _check_mc(answer_text, gt_answer)
        elif "binary" in question_type:
            is_correct = _check_binary(answer_text, gt_answer)
        elif "list" in question_type:
            is_correct = _check_list(answer_text, gt_answer, wrong_answer)
        else:
            # Fallback: try MC first (single letter), then binary, then exact
            norm = normalize_answer(answer_text)
            if len(gt_answer.strip()) == 1 and gt_answer.strip().isalpha():
                is_correct = _check_mc(answer_text, gt_answer)
            elif normalize_answer(gt_answer) in ("yes", "no"):
                is_correct = _check_binary(answer_text, gt_answer)
            else:
                is_correct = norm == normalize_answer(gt_answer)

        answer_score = answer_reward if is_correct else -answer_reward
        print(f"  Answer correct: {is_correct}")
    else:
        print("  [Content Validation] Skipped due to format errors or missing answer")

    total_score = format_score + answer_score
    print(f"  Format: {format_score}  Answer: {answer_score}  Total: {total_score}")
    print("=" * 80 + "\n")

    return total_score
