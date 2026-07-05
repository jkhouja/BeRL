"""
Prepare the DynToM benchmark (YangXiao-nlp/DynToM) for BeRL evaluation.

DynToM (Dynamic bucket): multi-scenario social stories where a character's mental
state evolves. Questions (type_a: state in a scenario, type_c: causal influence,
type_d: state trajectory across scenarios) are all multiple-choice.

HF: YangXiao-nlp/DynToM  (DynToM.json = dict of 1161 stories, ~83k questions total)
Each story: stage.story = {scenario N: {background, dialogue:[{speaker: utt}...]}},
            question = {qid: {question, "true answer"(letter), options:["a. ...",...]}}.

We sample a subset (default 2000) for a tractable eval.

Output: data/cleaned_tom/dyntom_test.parquet  (data_source = dyntom_<type>)

Usage:
    python examples/data_preprocess/prepare_dyntom.py \
        --out data/cleaned_tom/dyntom_test.parquet --limit 2000
"""

import argparse
import random

from huggingface_hub import hf_hub_download
import json

from tom_eval_common import make_messages, make_ground_truth, make_sample, write_parquet


def _render_story(story: dict) -> str:
    parts = []
    for i, key in enumerate(sorted(story.keys()), 1):
        sc = story[key]
        chunk = [f"Scenario {i}:"]
        if isinstance(sc, dict):
            if sc.get("background"):
                chunk.append(sc["background"])
            for turn in sc.get("dialogue", []) or []:
                if isinstance(turn, dict):
                    for speaker, utt in turn.items():
                        chunk.append(f"{speaker}: {utt}")
                else:
                    chunk.append(str(turn))
        else:
            chunk.append(str(sc))
        parts.append("\n".join(chunk))
    return "\n\n".join(parts)


def _parse_options(options):
    """Options like 'a. text' -> ordered list of (orig_letter, text)."""
    parsed = []
    for opt in options:
        opt = str(opt).strip()
        if len(opt) >= 2 and opt[1] in ".)" and opt[0].isalpha():
            parsed.append((opt[0].lower(), opt[2:].strip()))
        else:
            parsed.append((None, opt))
    return parsed


def build(limit=2000, seed=0):
    path = hf_hub_download("YangXiao-nlp/DynToM", "DynToM.json", repo_type="dataset")
    data = json.load(open(path))

    # Flatten to (story_context, question_dict) then sample.
    flat = []
    for story in data.values():
        context = _render_story(story.get("stage", {}).get("story", {}))
        for qid, q in story.get("question", {}).items():
            flat.append((context, q))
    rng = random.Random(seed)
    rng.shuffle(flat)
    if limit:
        flat = flat[:limit]

    rows, skipped = [], 0
    for i, (context, q) in enumerate(flat):
        gold = str(q.get("true answer", "")).strip().lower()
        options = q.get("options") or []
        parsed = _parse_options(options)
        letter_map = {ol: txt for ol, txt in parsed if ol}
        if gold not in letter_map:
            skipped += 1
            continue

        # Re-letter to uppercase A.. preserving option order.
        letter2text, lines, gold_upper = {}, [], None
        for j, (ol, txt) in enumerate(parsed):
            up = chr(ord("A") + j)
            letter2text[up] = txt
            lines.append(f"({up}) {txt}")
            if ol == gold:
                gold_upper = up
        if gold_upper is None:
            skipped += 1
            continue

        qtype = q.get("question type", "type")
        ds = f"dyntom_{qtype}"
        gt = make_ground_truth(gold_upper, "mc",
                               answer_text=letter2text[gold_upper], choices=letter2text)
        msgs = make_messages(context, q.get("question", ""), "\n".join(lines))
        rows.append(make_sample(
            ds, msgs, gt,
            question=q.get("question", ""), answer_text=letter2text[gold_upper], story=context,
            extra_info={"type": qtype},
        ))

    print(f"  (skipped {skipped} rows with unmatched gold letter)")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/cleaned_tom/dyntom_test.parquet")
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rows = build(limit=args.limit, seed=args.seed)
    write_parquet(rows, args.out)


if __name__ == "__main__":
    main()
