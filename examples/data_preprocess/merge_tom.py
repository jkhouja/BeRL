"""
Preprocess the merged ToM eval/train sets (Hi-ToM + ExploreToM + ToMi) to parquet.

Emits **model-agnostic** chat messages ``[{role: system}, {role: user}]`` (no baked chat-template
special tokens) so the tokenizer applies the correct template per-model at load time
(``data.prompt_is_text=False``). This is the same format as ``prepare_fantom.py`` /
``examples/data_preprocess/tom_eval_common.py``.

The system prompt is the ``cot_eval`` ToM prompt; ``--add_hint`` appends the room-witness
assumptions and ``--concise_answer`` appends the "output ONLY the key noun/object" instruction.
Running with ``--add_hint --concise_answer`` reproduces ``ToM_test_HiExTi_hint_v3.parquet`` (the
eval file used by the current training scripts).

Legacy note: this script previously baked a fully-rendered Qwen ChatML string into a single ``user``
message, which required ``data.prompt_is_text=True`` and double-wrapped under
``apply_chat_template``. That format was model-specific and is no longer produced.
"""

import os
import datasets
import argparse

import pandas as pd
from verl.utils.hdfs_io import copy, makedirs


# cot_eval system prompt (identical to prepare_fantom / tom_eval_common).
COT_EVAL_SYSTEM = (
    "You are a helpful assistant. The assistant first thinks about the reasoning process in the mind "
    "and then provides the user with the answer. The reasoning process and answer are enclosed within "
    "<think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>. "
    "Now the user asks you to solve a theory of mind reasoning problem. "
    "After thinking, when you finally reach a conclusion, clearly state your answer within "
    "<answer> </answer> tags."
)

# cot_eval variant without the tag-format instruction (used when --wo_think).
COT_EVAL_SYSTEM_WO_THINK = (
    "You are a helpful assistant. The assistant first thinks about the reasoning process in the mind "
    "and then provides the user with the answer. Now the user asks you to solve a theory of mind "
    "reasoning problem. Please reason step by step, and put your final answer within <answer> </answer> tags."
)

HINT_NOTE = (
    "\nNote: You should assume the following.\n"
    "(1) An agent witnesses everything and every movement before exiting a room.\n"
    "(2) An agent A can infer another agent B's mental state only if A and B have been "
    "in the same room, or have private or public interactions."
)

CONCISE_NOTE = (
    "\nImportant: In your <answer> tags, output ONLY the key noun or object "
    '(e.g., "kitchen", "red_box", "yes"), not a full sentence.'
)


def make_system_prompt(add_hint=False, wo_think=False, concise_answer=False):
    """Build the system prompt string for the requested variant."""
    system = COT_EVAL_SYSTEM_WO_THINK if wo_think else COT_EVAL_SYSTEM
    if add_hint:
        system += HINT_NOTE
    if concise_answer:
        system += CONCISE_NOTE
    return system


def make_messages(story, question, add_hint=False, wo_think=False, concise_answer=False):
    """Return model-agnostic ``[{system}, {user}]`` chat messages (template applied at load)."""
    quiz = f"Read the following story and answer the question. \nStory: {story}\nQuestion: {question}"
    return [
        {"role": "system", "content": make_system_prompt(add_hint, wo_think, concise_answer)},
        {"role": "user", "content": quiz},
    ]


def tom_make_map_fn():
    def process_fn(example, idx):
        messages = make_messages(
            story=example['story'],
            question=example['question'],
            add_hint=args.add_hint,
            wo_think=args.wo_think,
            concise_answer=args.concise_answer,
        )
        solution = example['answer']
        data_source = example['data_source']
        return {
            "data_source": data_source,
            "prompt": messages,
            "ability": "theory_of_mind",
            "reward_model": {
                "style": "rule",
                "ground_truth": solution
            },
            "extra_info": example.get('extra_info', {'key': 'dummy'}),
            "answer": solution,
            "question": example['question'],
            "story": example['story'],
        }
    return process_fn


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./data/cleaned_tom')
    parser.add_argument('--hdfs_dir', default=None)
    # random seed
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--add_hint', action='store_true',
                        help='Append the room-witness assumptions to the system prompt')
    parser.add_argument('--wo_think', action='store_true',
                        help='Use the system prompt variant without the <think>/<answer> tag-format instruction')
    parser.add_argument('--concise_answer', action='store_true',
                        help='Append the "output ONLY the key noun/object" instruction (the v3 eval prompt); '
                             'with --add_hint reproduces ToM_test_HiExTi_hint_v3.parquet')
    args = parser.parse_args()


    # deception,story_length,question_order,sample_id,story,question,choices,answer,question_old,answer_old
    # 3200 = 2000(hi_tom) + 1200(explore_tom)
    train_src = './data/cleaned_tom/merge/ToM_train_HiEx_3200.parquet'
    test_src = './data/cleaned_tom/merge/ToM_test_HiExTi.parquet'

    train_ds = datasets.load_dataset('parquet', data_files=train_src)['train']
    test_ds = datasets.load_dataset('parquet', data_files=test_src)['train']

    print('len of train_dataset:', len(train_ds))
    print('len of test_dataset:', len(test_ds))

    local_dir = os.path.expanduser(args.local_dir)
    hdfs_dir = args.hdfs_dir
    os.makedirs(local_dir, exist_ok=True)

    train_ds = train_ds.map(function=tom_make_map_fn(), with_indices=True)
    test_ds = test_ds.map(function=tom_make_map_fn(), with_indices=True)

    # Build the output filename from the active variant flags.
    suffix = ""
    if args.add_hint:
        suffix += "_hint"
    if args.wo_think:
        suffix += "_wo_think"
    if args.concise_answer:
        suffix += "_v3"
    train_ds.to_parquet(os.path.join(local_dir, f'ToM_train_HiEx{suffix}.parquet'))
    test_ds.to_parquet(os.path.join(local_dir, f'ToM_test_HiExTi{suffix}.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)

