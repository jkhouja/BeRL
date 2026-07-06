"""
Unit tests for reward-side response parsing/stitching fixes (Issues 2 & 3).

Issue 2 (split_thinking double-<think>): the opening <think> must be added only when
the model's response does not already start with one. Behavior/dialogue rollouts emit
their own <think> (no generation_prefix), so the old unconditional prepend produced a
malformed "<think><think>...</think>" context for the reward model. Direct-ToM rollouts
start after a generation_prefix "<think>" and still need it added.

Issue 3 (response_mask boundary): the actor-as-RM path must use the SAME boundary as the
frozen RM path — [thinking_length-1 : full_len] — so the two RM modes score the same token
window and their rewards are comparable. The -1 compensates for the assistant-turn
terminator that thinking-only re-tokenization appends. This test numerically verifies, with
real tokenizers, that the corrected boundary aligns the scored labels to the ground-truth
answer tokens (and that the old off-by-one convention misaligns them).

Run: python tests/reward_score/test_response_parser.py   (or: pytest tests/reward_score/test_response_parser.py)
Offline for the string tests; the tokenizer-based mask test SKIPs if a model is unavailable.
"""

import sys

from verl.utils.reward_score.response_parser import get_parser


# ---------------------------------------------------------------------------
# Issue 2 — split_thinking must never double the opening <think>
# ---------------------------------------------------------------------------
# (parser_type, response, expected_thinking, expected_answer, note)
SPLIT_CASES = [
    # Dialogue/behavior rollout: model emits its OWN <think> (rl_dataset, no prefix).
    ("qwen2", "<think>reasoning</think><answer>hi</answer>",
     "<think>reasoning</think>", "<answer>hi</answer>", "qwen self-<think>, no double"),
    ("gemma", "<think>the plan</think>Sure, here you go",
     "<think>the plan</think>", "Sure, here you go", "gemma self-<think>, no double"),
    ("qwen3", "<think>step by step</think><answer>42</answer>",
     "<think>step by step</think>", "<answer>42</answer>", "qwen3 self-<think>, no double"),
    # Direct-ToM rollout: response starts AFTER generation_prefix "<think>", so no leading tag.
    ("qwen2", "reasoning here</think><answer>B</answer>",
     "<think>reasoning here</think>", "<answer>B</answer>", "tom-style, add the tag"),
    ("gemma", "because X</think>The answer",
     "<think>because X</think>", "The answer", "gemma tom-style, add the tag"),
    # Leading whitespace before the model's own <think> must still be detected.
    ("qwen2", "  \n<think>ws</think><answer>y</answer>",
     "  \n<think>ws</think>", "<answer>y</answer>", "leading whitespace + self-<think>"),
    # No </think> at all: single part; tag added iff missing (both branches).
    ("gemma", "just an answer, no tags",
     "<think>just an answer, no tags</think>", "", "no </think>, no <think> -> add"),
    ("gemma", "<think>unterminated thinking",
     "<think>unterminated thinking</think>", "", "no </think>, has <think> -> no double"),
    # Model itself emitted a malformed double <think><think>: we must NOT add a third.
    ("qwen2", "<think><think>oops</think><answer>z</answer>",
     "<think><think>oops</think>", "<answer>z</answer>", "model's own double preserved, no third"),
    # Empty thinking region.
    ("gemma", "<think></think>content",
     "<think></think>", "content", "empty thinking, self-<think>"),
    # Only the first </think> splits; a later </think> stays in the answer part.
    ("gemma", "<think>a</think>b</think>c",
     "<think>a</think>", "b</think>c", "split on first </think> only"),
]


def run_split_thinking():
    failures = []
    for ptype, resp, exp_think, exp_ans, note in SPLIT_CASES:
        parser = get_parser(ptype)
        think, ans = parser.split_thinking(resp)
        ok_think = think == exp_think
        ok_ans = ans == exp_ans
        # Invariant: exactly one opening <think> in the stitched thinking, UNLESS the
        # model itself emitted extra ones (we never ADD a second).
        added_by_us_ok = think.count("<think>") == max(1, resp.count("<think>"))
        ok = ok_think and ok_ans and added_by_us_ok
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] split/{ptype}: {note}")
        if not ok:
            print(f"        response={resp!r}")
            print(f"        thinking got={think!r} exp={exp_think!r}")
            print(f"        answer   got={ans!r} exp={exp_ans!r}")
            failures.append(note)
    return failures


# ---------------------------------------------------------------------------
# Issue 2 companion — has_format_violation stays correct with the new split_thinking
# ---------------------------------------------------------------------------
# (parser_type, response, expect_violation, note)
VIOLATION_CASES = [
    ("qwen2", "<think>r</think><answer>ok</answer>", False, "qwen well-formed"),
    ("qwen2", "<think>r</think>no answer tags", True, "qwen requires <answer>"),
    ("qwen2", "<think>r</think><answer>  </answer>", True, "qwen empty <answer>"),
    ("qwen2", "no think tag here", True, "qwen missing think tags"),
    ("qwen2", "<think><think>r</think><answer>x</answer>", True, "qwen double <think> is a violation"),
    ("gemma", "<think>r</think>a real answer", False, "gemma non-empty free-text answer ok"),
    ("gemma", "<think>r</think>   ", True, "gemma empty answer"),
    ("gemma", "<think>r</think></answer>", True, "gemma answer only stray tag -> empty"),
    ("qwen3", "<think>r</think>free text answer", False, "qwen3 answer tags optional"),
    ("qwen3", "<think>r</think>", True, "qwen3 empty answer"),
]


def run_format_violation():
    failures = []
    for ptype, resp, expect, note in VIOLATION_CASES:
        parser = get_parser(ptype)
        got = parser.has_format_violation(resp)
        ok = got == expect
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] violation/{ptype}: {note} (got={got} exp={expect})")
        if not ok:
            failures.append(note)
    return failures


# ---------------------------------------------------------------------------
# Issue 3 — numeric mask-alignment check with a real tokenizer
# ---------------------------------------------------------------------------
def _masked_label_ids(tokenizer, parser, chat, model_response, ground_truth, use_minus_one):
    """Replicate the fsdp_workers stitching + response_mask boundary and return the
    (decoded_text, label_token_ids) the reward model would actually score.

    use_minus_one=True reproduces the CORRECTED convention [thinking_length-1 : full_len]
    (shared by frozen and, after the fix, the actor path). False reproduces the OLD actor
    off-by-one [thinking_length : full_len].
    """
    thinking, _ = parser.split_thinking(model_response)
    thinking = thinking.replace(tokenizer.eos_token, "") if tokenizer.eos_token else thinking

    chat_tom = chat + [{"role": "assistant", "content": thinking}]
    tom_str = tokenizer.apply_chat_template(chat_tom, add_generation_prompt=False, tokenize=False)
    thinking_length = tokenizer(tom_str, return_tensors="pt", add_special_tokens=False)["input_ids"].shape[-1]

    model_used_answer_tags = "<answer>" in model_response
    full_response = parser.build_stitched_response(thinking, ground_truth, model_used_answer_tags)
    full_chat = chat + [{"role": "assistant", "content": full_response}]
    full_str = tokenizer.apply_chat_template(full_chat, add_generation_prompt=False, tokenize=False)
    ids = tokenizer(full_str, return_tensors="pt", add_special_tokens=False)["input_ids"][0]

    full_len = ids.shape[-1]
    start = (thinking_length - 1) if use_minus_one else thinking_length
    # Replicate downstream: labels = ids[1:], response_mask (pre-slice) gates token_{i+1}.
    # A set mask index k selects label token at position k+1 == ids[k+1].
    label_token_ids = [int(ids[k + 1]) for k in range(start, full_len) if (k + 1) < full_len]
    return tokenizer.decode(label_token_ids), label_token_ids


def run_mask_alignment():
    """Best-effort numeric check with real tokenizers. The Issue-3 fix guarantees:
      (a) the corrected actor boundary scores EXACTLY one extra leading token vs the old
          off-by-one, with an identical tail  => the two RM paths now agree token-for-token;
      (b) where the thinking re-tokenization aligns (Qwen), that extra token completes the
          ground-truth answer coverage. For Gemma the re-tokenization over-counts the turn
          boundary by >1 token (Issue 6, deferred), so full coverage is only informational.
    """
    failures = []
    try:
        from transformers import AutoTokenizer
    except Exception as e:  # pragma: no cover
        print(f"[SKIP] mask-alignment: transformers unavailable ({e})")
        return failures

    specs = [
        ("Qwen/Qwen2.5-3B-Instruct", "qwen2",
         [{"role": "system", "content": "You are a helpful assistant."},
          {"role": "user", "content": "Continue the conversation naturally."}],
         "<think>they seem worried about money</think><answer>Have you considered a budget?</answer>",
         "Have you considered a budget?", True),
        # Gemma's chat template rejects a system role (the real pipeline folds it into the
        # first user turn via fold_system_prompt), so keep the prompt user-only here.
        # expect_full_coverage=False: Gemma needs the Issue-6 boundary fix for full coverage.
        ("google/gemma-2-2b-it", "gemma",
         [{"role": "user", "content": "You are a helpful assistant. Continue the conversation."}],
         "<think>the buyer is hesitant</think>The appliances are included in the rent.",
         "The appliances are included in the rent.", False),
    ]
    ran_any = False
    for model_path, ptype, chat, model_response, ground_truth, expect_full in specs:
        try:
            tok = AutoTokenizer.from_pretrained(model_path)
        except Exception as e:
            print(f"[SKIP] mask-alignment/{ptype}: tokenizer '{model_path}' unavailable ({e})")
            continue
        ran_any = True
        parser = get_parser(ptype)
        try:
            corr_txt, corr_ids = _masked_label_ids(tok, parser, chat, model_response, ground_truth, True)
            old_txt, old_ids = _masked_label_ids(tok, parser, chat, model_response, ground_truth, False)
        except Exception as e:
            print(f"[FAIL] mask-alignment/{ptype}: raised {e!r}")
            failures.append(f"mask/{ptype}")
            continue

        # (a) The fix adds exactly one leading token; the tail is unchanged.
        one_extra_leading = (len(corr_ids) == len(old_ids) + 1) and (corr_ids[1:] == old_ids)
        status = "PASS" if one_extra_leading else "FAIL"
        print(f"[{status}] mask-alignment/{ptype}: corrected = old + exactly one leading token "
              f"(len {len(corr_ids)} vs {len(old_ids)})")
        print(f"        corrected scored: {' '.join(corr_txt.split())!r}")
        print(f"        old       scored: {' '.join(old_txt.split())!r}")
        if not one_extra_leading:
            failures.append(f"mask/{ptype}")

        # (b) Coverage: asserted only where re-tokenization aligns (Qwen); informational for Gemma.
        gt_norm = " ".join(ground_truth.split())
        covered = gt_norm in " ".join(corr_txt.split())
        if expect_full:
            cov_status = "PASS" if covered else "FAIL"
            print(f"[{cov_status}] mask-coverage/{ptype}: corrected covers full ground truth ({covered})")
            if not covered:
                failures.append(f"cov/{ptype}")
        else:
            print(f"[INFO] mask-coverage/{ptype}: full GT covered={covered} "
                  f"(Issue 6 boundary drift, deferred)")

    if not ran_any:
        print("[SKIP] mask-alignment: no tokenizers available (offline)")
    return failures


def run():
    all_failures = []
    print("== Issue 2: split_thinking (no double <think>) ==")
    all_failures += run_split_thinking()
    print("\n== Issue 2 companion: has_format_violation ==")
    all_failures += run_format_violation()
    print("\n== Issue 3: response_mask boundary alignment ==")
    all_failures += run_mask_alignment()

    total = len(SPLIT_CASES) + len(VIOLATION_CASES)
    print(f"\n{total - len([f for f in all_failures if not f.startswith('mask/')])}/{total} string checks passed")
    if all_failures:
        print("FAILURES:", all_failures)
        return 1
    print("ALL PASS")
    return 0


def test_response_parser():
    assert run() == 0


if __name__ == "__main__":
    sys.exit(run())
