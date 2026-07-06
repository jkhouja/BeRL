"""
Unit tests for reward-side response parsing/stitching fixes (Issues 2, 3, 5 & 6).

Issue 2 (split_thinking double-<think>): the opening <think> must be added only when
the model's response does not already start with one. Behavior/dialogue rollouts emit
their own <think> (no generation_prefix), so the old unconditional prepend produced a
malformed "<think><think>...</think>" context for the reward model. Direct-ToM rollouts
start after a generation_prefix "<think>" and still need it added.

Issue 3 / Issue 6 (response_mask boundary): the mask that selects the ground-truth answer
tokens for scoring must (a) be identical between the actor-as-RM and frozen RM paths, and
(b) cover the FULL ground-truth answer for every model family. The boundary is now found via
the tokenizer's character offset mapping (ModelResponseParser.answer_token_start), robust to
model-specific turn terminators — the old re-tokenised thinking_length minus a fixed -1
under-covered the answer on Gemma (multi-token <end_of_turn>) by >1 token. This test verifies
full coverage and no thinking-text leakage with real tokenizers.

Issue 5 (relative invalid sentinel): invalid / unscoreable rollouts must always rank below the
worst valid response, for every reward_type. A fixed -40 sentinel overlapped the negative valid
range of log_prob / neg_perplexity; the sentinel is now computed relative to the reward-type's
valid floor with a large fixed gap. This test verifies the ordering numerically.

Run: python tests/reward_score/test_response_parser.py   (or: pytest tests/reward_score/test_response_parser.py)
The string/numeric tests run offline; the tokenizer-based mask test SKIPs if a model is unavailable.
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
# Issue 6 — exact answer-region boundary via offset mapping (real tokenizer)
# ---------------------------------------------------------------------------
def _scored_region(tokenizer, parser, chat, model_response, ground_truth):
    """Replicate the fsdp_workers stitching + the NEW offset-mapping boundary and return
    (scored_text, n_scored_tokens, thinking_text) the reward model would actually score.

    Mirrors RewardModelWorker._switch_chat_template / _build_actor_rm_inputs: the response
    mask is set from `answer_token_start - 1` (the -1 aligns with the downstream labels =
    ids[1:] / response_mask = mask[:-1] slice), so a set index k scores label ids[k+1].
    """
    thinking, _ = parser.split_thinking(model_response)
    thinking = thinking.replace(tokenizer.eos_token, "") if tokenizer.eos_token else thinking

    chat_tom = chat + [{"role": "assistant", "content": thinking}]
    tom_str = tokenizer.apply_chat_template(chat_tom, add_generation_prompt=False, tokenize=False)
    thinking_length = tokenizer(tom_str, return_tensors="pt", add_special_tokens=False)["input_ids"].shape[-1]

    model_used_answer_tags = "<answer>" in model_response
    full_response = parser.build_stitched_response(thinking, ground_truth, model_used_answer_tags)
    full_str = tokenizer.apply_chat_template(chat + [{"role": "assistant", "content": full_response}],
                                             add_generation_prompt=False, tokenize=False)
    ids = tokenizer(full_str, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    full_len = ids.shape[-1]

    answer_start = parser.answer_token_start(tokenizer, full_str, thinking_length)
    start = answer_start - 1
    label_token_ids = [int(ids[k + 1]) for k in range(start, full_len) if (k + 1) < full_len]
    return tokenizer.decode(label_token_ids), len(label_token_ids), thinking


def run_issue6_mask():
    """The offset-mapping boundary must cover the FULL ground-truth answer for every model
    family (Qwen single-token terminator AND Gemma multi-token terminator), without leaking
    the model's thinking text into the scored region (beyond a single BPE-glued boundary
    token). Real tokenizers required; SKIPs cleanly when unavailable.
    """
    failures = []
    try:
        from transformers import AutoTokenizer
    except Exception as e:  # pragma: no cover
        print(f"[SKIP] issue6-mask: transformers unavailable ({e})")
        return failures

    specs = [
        ("Qwen/Qwen2.5-3B-Instruct", "qwen2",
         [{"role": "system", "content": "You are a helpful assistant."},
          {"role": "user", "content": "Continue the conversation naturally."}],
         "<think>they seem worried about money</think><answer>Have you considered a budget?</answer>",
         "Have you considered a budget?", "worried about money"),
        # Gemma's chat template rejects a system role (the real pipeline folds it into the
        # first user turn), so keep the prompt user-only here.
        ("google/gemma-2-2b-it", "gemma",
         [{"role": "user", "content": "You are a helpful assistant. Continue the conversation."}],
         "<think>the buyer is hesitant</think>The appliances are included in the rent.",
         "The appliances are included in the rent.", "the buyer is hesitant"),
    ]
    ran_any = False
    for model_path, ptype, chat, model_response, ground_truth, thinking_probe in specs:
        try:
            tok = AutoTokenizer.from_pretrained(model_path)
        except Exception as e:
            print(f"[SKIP] issue6-mask/{ptype}: tokenizer '{model_path}' unavailable ({e})")
            continue
        ran_any = True
        parser = get_parser(ptype)
        try:
            scored_txt, n_scored, thinking = _scored_region(tok, parser, chat, model_response, ground_truth)
        except Exception as e:
            print(f"[FAIL] issue6-mask/{ptype}: raised {e!r}")
            failures.append(f"issue6/{ptype}")
            continue

        norm = lambda s: " ".join(s.split())
        covered = norm(ground_truth) in norm(scored_txt)
        # thinking text (minus the tags) must NOT be scored (allow the single glued boundary token)
        think_body = norm(thinking.replace(parser.THINK_OPEN, "").replace(parser.THINK_CLOSE, ""))
        leaks = think_body and think_body in norm(scored_txt)

        cov_status = "PASS" if covered else "FAIL"
        print(f"[{cov_status}] issue6-coverage/{ptype}: covers full ground truth ({covered})")
        print(f"        scored: {norm(scored_txt)!r}")
        if not covered:
            failures.append(f"issue6-cov/{ptype}")

        leak_status = "PASS" if not leaks else "FAIL"
        print(f"[{leak_status}] issue6-noleak/{ptype}: thinking body not scored ({not leaks})")
        if leaks:
            failures.append(f"issue6-leak/{ptype}")

    if not ran_any:
        print("[SKIP] issue6-mask: no tokenizers available (offline)")
    return failures


# ---------------------------------------------------------------------------
# Issue 5 — relative invalid sentinel always ranks below the worst valid response
# ---------------------------------------------------------------------------
def _reward_tail(raw_scores, invalid_mask, token_counts, reward_type,
                 format_penalty=0.0, format_viol=None):
    """Pure re-implementation of the fsdp_workers reward tail (Issue 5) for testing:
    valid clamp -> graded format penalty -> relative invalid sentinel -> nan/0-token
    handling -> final clamp. Kept byte-for-byte consistent with _actor_rm_forward /
    _forward_micro_batch so the ordering guarantee is what actually ships.
    """
    import torch
    from verl.workers.fsdp_workers import (valid_reward_floor, invalid_reward_value,
                                           MAX_REWARD)
    rm = raw_scores.clone().float()
    invalid_value = invalid_reward_value(reward_type, format_penalty)
    rm = torch.clamp(rm, min=valid_reward_floor(reward_type), max=MAX_REWARD)
    if format_penalty and format_viol is not None:
        rm = rm - format_viol.float() * format_penalty
    rm = rm.masked_fill(invalid_mask, invalid_value)
    rm = torch.where(token_counts > 0, rm, torch.full_like(rm, invalid_value))
    rm = torch.nan_to_num(rm, nan=invalid_value, posinf=float(MAX_REWARD), neginf=invalid_value)
    rm = torch.clamp(rm, min=invalid_value, max=MAX_REWARD)
    return rm, invalid_value


def run_invalid_sentinel():
    """For every reward_type the invalid sentinel must be strictly below EVERY valid score,
    including a maximally format-penalised valid one, and NaN / zero-token rollouts must map
    onto the sentinel (not into the valid range). This is the Issue-5 guarantee GRPO relies
    on to rank malformed rollouts below poor-but-valid ones.
    """
    import math
    import torch
    failures = []

    # (reward_type, valid raw post-transform scores spanning the plausible range)
    scenarios = {
        "power": [0.0, 0.01, 0.5, 4.0, 39.0, 100.0],                 # >=0; 100 clamps to 40
        "log_prob": [-50.0, -12.0, -4.0, -1.0, -0.05],               # avg log prob (<=0)
        "neg_perplexity": [-math.exp(6), -math.exp(2), -math.exp(0.3), -1.0],  # -exp(-ll)
    }
    for rtype, valids in scenarios.items():
        for fp in (0.0, 5.0):
            n = len(valids)
            raw = torch.tensor(valids + [-999.0, float("nan"), 0.0], dtype=torch.float32)
            invalid_mask = torch.tensor([False] * n + [True, False, False])
            token_counts = torch.tensor([5] * n + [5, 5, 0])  # last: zero-token -> invalid
            fviol = torch.zeros(n + 3)
            out, inv = _reward_tail(raw, invalid_mask, token_counts, rtype, fp, fviol)
            valid_out = out[:n]
            invalid_out = out[n:]  # explicit-invalid, NaN, zero-token
            ok_sep = bool((invalid_out.max() < valid_out.min()).item())
            ok_all_sentinel = bool(torch.allclose(invalid_out, torch.full_like(invalid_out, inv)))
            ok = ok_sep and ok_all_sentinel
            status = "PASS" if ok else "FAIL"
            print(f"[{status}] sentinel/{rtype} fp={fp}: invalid.max={invalid_out.max():.3f} "
                  f"< valid.min={valid_out.min():.3f} (sentinel={inv:.1f})")
            if not ok:
                failures.append(f"sentinel/{rtype}/fp{fp}")

        # malformed-but-valid (format_violation=1) must still rank above hard-invalid
        if True:
            n = len(valids)
            raw = torch.tensor(valids + [-999.0], dtype=torch.float32)
            invalid_mask = torch.tensor([False] * n + [True])
            token_counts = torch.tensor([5] * (n + 1))
            fviol = torch.tensor([1.0] * n + [0.0])  # all valids malformed
            out, inv = _reward_tail(raw, invalid_mask, token_counts, rtype, 5.0, fviol)
            ok = bool((out[n] < out[:n].min()).item())
            status = "PASS" if ok else "FAIL"
            print(f"[{status}] sentinel/{rtype} malformed-valid>invalid: "
                  f"invalid={out[n]:.3f} < worst malformed-valid={out[:n].min():.3f}")
            if not ok:
                failures.append(f"sentinel/{rtype}/malformed")
    return failures


def run():
    all_failures = []
    print("== Issue 2: split_thinking (no double <think>) ==")
    all_failures += run_split_thinking()
    print("\n== Issue 2 companion: has_format_violation ==")
    all_failures += run_format_violation()
    print("\n== Issue 5: relative invalid sentinel ==")
    all_failures += run_invalid_sentinel()
    print("\n== Issue 6: exact answer-region boundary (offset mapping) ==")
    all_failures += run_issue6_mask()

    total = len(SPLIT_CASES) + len(VIOLATION_CASES)
    string_fail = len([f for f in all_failures
                       if not f.startswith(("issue6/", "issue6-", "sentinel/"))])
    print(f"\n{total - string_fail}/{total} string checks passed")
    if all_failures:
        print("FAILURES:", all_failures)
        return 1
    print("ALL PASS")
    return 0


def test_response_parser():
    assert run() == 0


if __name__ == "__main__":
    sys.exit(run())
