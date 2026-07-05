"""
Preprocess the FANToM dataset for ToM evaluation.
Downloads fantom.tar.gz, extracts fantom_v1.json, and produces a parquet
with the same schema used by merge_tom.py (prompt / data_source / ability / reward_model / extra_info).

Supported question types:
  - belief MC        (fantom_belief_mc)
  - answerability binary (fantom_answerability_binary)
  - answerability list   (fantom_answerability_list)
  - info-access binary   (fantom_info_binary)
  - info-access list     (fantom_info_list)
"""

import os
import json
import random
import tarfile
import argparse
import tempfile
import urllib.request

import json as _json
import pandas as pd


# ---------------------------------------------------------------------------
# Prompt construction (mirrors merge_tom.py make_prefix)
# ---------------------------------------------------------------------------

def make_prefix(context, question, template_type='messages', add_hint=False, wo_think=False):
    """Build the prompt as a clean list of chat messages.

    The messages are stored WITHOUT any model-specific special tokens
    (e.g. ``<|im_start|>``). The chat template for the configured model is
    applied later at load time in ``RLHFDataset`` so the same parquet works
    for Qwen2.5, Qwen3, Gemma2, etc. ``template_type`` is accepted for
    backward compatibility but ignored.
    """
    quiz = f"{context}\n\nQuestion: {question}"

    hint = ""
    if add_hint:
        hint = (
            "\nNote: You should assume the following.\n"
            "(1) An agent witnesses everything and every movement before exiting a room.\n"
            "(2) An agent A can infer another agent B's mental state only if A and B have been in the same room, "
            "or have private or public interactions.\n"
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
# Download & extract
# ---------------------------------------------------------------------------

FANTOM_URL = "https://storage.googleapis.com/ai2-mosaic-public/projects/fantom/fantom.tar.gz"


def download_fantom(cache_dir):
    """Download and extract fantom_v1.json, return path."""
    json_path = os.path.join(cache_dir, "fantom_v1.json")
    if os.path.exists(json_path):
        print(f"Using cached {json_path}")
        return json_path

    os.makedirs(cache_dir, exist_ok=True)
    tar_path = os.path.join(cache_dir, "fantom.tar.gz")
    if not os.path.exists(tar_path):
        print(f"Downloading FANToM from {FANTOM_URL} ...")
        urllib.request.urlretrieve(FANTOM_URL, tar_path)

    print("Extracting ...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(cache_dir)

    # The json may be nested under a subdirectory
    for root, dirs, files in os.walk(cache_dir):
        for f in files:
            if f == "fantom_v1.json":
                found = os.path.join(root, f)
                if found != json_path:
                    os.rename(found, json_path)
                return json_path

    raise FileNotFoundError("fantom_v1.json not found in archive")


# ---------------------------------------------------------------------------
# MC belief: shuffle choices
# ---------------------------------------------------------------------------

def make_mc_options(correct, wrong):
    """Return (formatted_choices_str, correct_letter)."""
    options = [(correct, True), (wrong, False)]
    random.shuffle(options)
    letters = ["A", "B"]
    correct_letter = None
    lines = []
    for letter, (text, is_correct) in zip(letters, options):
        lines.append(f"({letter}) {text}")
        if is_correct:
            correct_letter = letter
    return "\n".join(lines), correct_letter


# ---------------------------------------------------------------------------
# Build samples
# ---------------------------------------------------------------------------

def build_samples(data, context_type, template_type, add_hint, wo_think):
    samples = []

    def _to_str(val):
        """Convert list or other types to comma-separated string."""
        if isinstance(val, list):
            return ", ".join(str(v) for v in val)
        return str(val) if val else ""

    def _make_sample(data_source, prompt_text, gt_json, question_text, answer_text, context, extra_info):
        """Build a sample dict with all required columns for schema compatibility.

        ``prompt_text`` is a list of chat messages (model-agnostic); the chat
        template is applied later at load time based on the active model.
        """
        return {
            "data_source": data_source,
            "prompt": prompt_text,
            "ability": "theory_of_mind",
            "reward_model": {"style": "rule", "ground_truth": gt_json},
            "extra_info": extra_info,
            "answer": answer_text,
            "question": question_text,
            "story": context,
        }

    for entry in data:
        set_id = entry["set_id"]
        context = entry.get(f"{context_type}_context", entry.get("short_context", ""))

        # --- Belief MC ---
        for qa in entry.get("beliefQAs", []):
            question = qa["question"]
            correct = qa["correct_answer"]
            wrong = qa.get("wrong_answer", "")
            choices_str, correct_letter = make_mc_options(correct, wrong)
            full_question = f"{question}\n{choices_str}"
            prompt_text = make_prefix(context, full_question, template_type, add_hint, wo_think)
            samples.append(_make_sample(
                "fantom_belief_mc", prompt_text,
                _json.dumps({"answer": correct_letter, "question_type": "belief_mc", "wrong_answer": ""}),
                full_question, correct_letter, context,
                {"set_id": set_id, "question_type": "belief_mc", "context_type": context_type,
                 "correct_answer_text": correct, "wrong_answer": wrong},
            ))

        # --- Answerability binary ---
        for qa in entry.get("answerabilityQAs_binary", []):
            question = qa["question"]
            correct = qa["correct_answer"]  # "yes" or "no"
            full_question = f"{question} Answer yes or no."
            prompt_text = make_prefix(context, full_question, template_type, add_hint, wo_think)
            samples.append(_make_sample(
                "fantom_answerability_binary", prompt_text,
                _json.dumps({"answer": correct, "question_type": "answerability_binary", "wrong_answer": ""}),
                full_question, correct, context,
                {"set_id": set_id, "question_type": "answerability_binary", "context_type": context_type,
                 "correct_answer_text": "", "wrong_answer": ""},
            ))

        # --- Answerability list ---
        ans_list_qa = entry.get("answerabilityQA_list")
        if ans_list_qa:
            # The list question asks which characters can answer a target question
            target_q = ans_list_qa.get("target_question", "")
            question = ans_list_qa["question"]
            correct = _to_str(ans_list_qa["correct_answer"])
            wrong = _to_str(ans_list_qa.get("wrong_answer", ""))
            full_question = f"Target: {target_q}\n{question}"
            prompt_text = make_prefix(context, full_question, template_type, add_hint, wo_think)
            samples.append(_make_sample(
                "fantom_answerability_list", prompt_text,
                _json.dumps({"answer": correct, "question_type": "answerability_list", "wrong_answer": wrong}),
                full_question, correct, context,
                {"set_id": set_id, "question_type": "answerability_list", "context_type": context_type,
                 "correct_answer_text": "", "wrong_answer": wrong},
            ))

        # --- Info-access binary ---
        for qa in entry.get("infoAccessibilityQAs_binary", []):
            question = qa["question"]
            correct = qa["correct_answer"]
            full_question = f"{question} Answer yes or no."
            prompt_text = make_prefix(context, full_question, template_type, add_hint, wo_think)
            samples.append(_make_sample(
                "fantom_info_binary", prompt_text,
                _json.dumps({"answer": correct, "question_type": "info_binary", "wrong_answer": ""}),
                full_question, correct, context,
                {"set_id": set_id, "question_type": "info_binary", "context_type": context_type,
                 "correct_answer_text": "", "wrong_answer": ""},
            ))

        # --- Info-access list ---
        info_list_qa = entry.get("infoAccessibilityQA_list")
        if info_list_qa:
            info_q = info_list_qa.get("information", "")
            question = info_list_qa["question"]
            correct = _to_str(info_list_qa["correct_answer"])
            wrong = _to_str(info_list_qa.get("wrong_answer", ""))
            full_question = f"Information: {info_q}\n{question}"
            prompt_text = make_prefix(context, full_question, template_type, add_hint, wo_think)
            samples.append(_make_sample(
                "fantom_info_list", prompt_text,
                _json.dumps({"answer": correct, "question_type": "info_list", "wrong_answer": wrong}),
                full_question, correct, context,
                {"set_id": set_id, "question_type": "info_list", "context_type": context_type,
                 "correct_answer_text": "", "wrong_answer": wrong},
            ))

    return samples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare FANToM eval parquet")
    parser.add_argument("--local_dir", default="./data/cleaned_tom")
    parser.add_argument("--template_type", type=str, default="messages",
                        choices=["messages", "base", "qwen-instruct", "dpsk-reasoning"],
                        help="Kept for backward compatibility; prompts are always stored as model-agnostic chat messages.")
    parser.add_argument("--add_hint", action="store_true")
    parser.add_argument("--wo_think", action="store_true")
    parser.add_argument("--context_type", type=str, default="short",
                        choices=["short", "full"],
                        help="Use short or full conversation context")
    parser.add_argument("--cache_dir", default="./data/fantom_cache",
                        help="Directory to cache downloaded FANToM files")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    json_path = download_fantom(args.cache_dir)
    with open(json_path, "r") as f:
        fantom_data = json.load(f)

    print(f"Loaded {len(fantom_data)} FANToM sets")

    samples = build_samples(
        fantom_data, args.context_type, args.template_type, args.add_hint, args.wo_think
    )

    df = pd.DataFrame(samples)
    os.makedirs(args.local_dir, exist_ok=True)
    out_path = os.path.join(args.local_dir, "fantom_test.parquet")

    # Use datasets library for parquet serialization (handles nested dicts consistently)
    import datasets
    ds = datasets.Dataset.from_pandas(df)
    ds.to_parquet(out_path)

    print(f"\nSaved {len(df)} samples to {out_path}")
    print("Breakdown by data_source:")
    print(df["data_source"].value_counts().to_string())
