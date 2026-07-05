"""
Shared helpers for building ToM *evaluation* parquets.

All eval prep scripts (prepare_simpletom.py, prepare_tombench.py, prepare_bigtom.py,
prepare_opentom.py, prepare_mmlu.py, ...) emit the SAME row schema that the training
pipeline / FANToM prep use, so a single reward_score dispatch handles them:

    {
        "data_source": str,          # unique tag routed in main_ppo._select_rm_score_fn
        "prompt":      [ {role, content}, ... ],   # model-agnostic chat messages
        "ability":     "theory_of_mind",
        "reward_model": {"style": "rule", "ground_truth": <json str>},
        "extra_info":  {...},
        "answer":      str,          # human-readable gold answer (debug)
        "question":    str,
        "story":       str,
    }

``ground_truth`` is a JSON string understood by verl.utils.reward_score.tom_mc:

    {"answer": "<letter | yes/no | text>",
     "answer_text": "<gold option text>",
     "question_type": "mc" | "binary" | "list" | "exact",
     "wrong_answer": "<'; '-joined distractor names, list qtype only>",
     "choices": {"A": "...", "B": "..."}}

The system prompt is IDENTICAL to prepare_fantom / the ToM eval prompt (cot_eval
style) so eval matches the training/eval convention used across BeRL.
"""

import json
import os
import random
from typing import Dict, List, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Prompt construction (mirrors examples/data_preprocess/prepare_fantom.make_prefix)
# ---------------------------------------------------------------------------

def make_messages(context: str, question: str, choices_block: str = "",
                  add_hint: bool = False, wo_think: bool = False) -> List[Dict[str, str]]:
    """Build model-agnostic chat messages (no special tokens; template applied at load)."""
    quiz = f"{context}\n\nQuestion: {question}"
    if choices_block:
        quiz = f"{quiz}\n{choices_block}"

    hint = ""
    if add_hint:
        hint = (
            "\nNote: You should assume the following.\n"
            "(1) An agent witnesses everything and every movement before exiting a room.\n"
            "(2) An agent A can infer another agent B's mental state only if A and B have been "
            "in the same room, or have private or public interactions.\n"
        )

    if not wo_think:
        system_content = (
            "You are a helpful assistant. The assistant first thinks about the reasoning process in the mind "
            "and then provides the user with the answer. The reasoning process and answer are enclosed within "
            "<think> </think> and <answer> </answer> tags, respectively, i.e., "
            "<think> reasoning process here </think><answer> answer here </answer>. "
            "Now the user asks you to solve a theory of mind reasoning problem. "
            "After thinking, when you finally reach a conclusion, clearly state your answer within "
            f"<answer> </answer> tags.\n{hint}"
        )
    else:
        system_content = (
            "You are a helpful assistant. The assistant first thinks about the reasoning process in the mind "
            "and then provides the user with the answer. Now the user asks you to solve a theory of mind "
            f"reasoning problem. Please reason step by step, and put your final answer within <answer> </answer> tags.\n{hint}"
        )

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": quiz},
    ]


# ---------------------------------------------------------------------------
# Multiple-choice option formatting
# ---------------------------------------------------------------------------

def format_choices(options: List[str], shuffle: bool = False, seed: Optional[int] = None):
    """Format a list of option texts as '(A) ...\\n(B) ...' with letters.

    Returns (choices_block, letter2text: dict).
    """
    idx = list(range(len(options)))
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(idx)
    letters = [chr(ord("A") + i) for i in range(len(options))]
    lines, letter2text = [], {}
    for letter, oi in zip(letters, idx):
        lines.append(f"({letter}) {options[oi]}")
        letter2text[letter] = options[oi]
    return "\n".join(lines), letter2text


# ---------------------------------------------------------------------------
# Ground-truth + row builders
# ---------------------------------------------------------------------------

def make_ground_truth(answer: str, question_type: str = "mc", answer_text: str = "",
                      wrong_answer: str = "", choices: Optional[Dict[str, str]] = None) -> str:
    """Serialize the ground-truth dict consumed by reward_score.tom_mc."""
    return json.dumps({
        "answer": str(answer),
        "answer_text": str(answer_text),
        "question_type": question_type,
        "wrong_answer": str(wrong_answer),
        "choices": choices or {},
    })


def make_sample(data_source: str, messages: List[Dict[str, str]], gt_json: str,
                question: str, answer_text: str, story: str, extra_info: Optional[dict] = None) -> dict:
    """Build one eval row with the schema shared across BeRL eval parquets."""
    return {
        "data_source": data_source,
        "prompt": messages,
        "ability": "theory_of_mind",
        "reward_model": {"style": "rule", "ground_truth": gt_json},
        "extra_info": extra_info or {},
        "answer": answer_text,
        "question": question,
        "story": story,
    }


def write_parquet(rows: List[dict], out_path: str):
    """Write rows to parquet, creating parent dirs. Prints a short summary."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(out_path, index=False)
    print(f"  ✓ wrote {len(df)} rows -> {out_path}")
    if "data_source" in df.columns:
        print("  per-source counts:")
        for src, n in df["data_source"].value_counts().items():
            print(f"    {src}: {n}")
    return out_path
