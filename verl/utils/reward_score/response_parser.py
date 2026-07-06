"""
Model-specific response parsing abstraction.

Encapsulates the logic for extracting assistant responses, answers, thinking sections,
and special tokens for different model families (Qwen2, Qwen3, Gemma2, etc.).
"""

import re
from typing import Optional, Tuple


class ModelResponseParser:
    """Base class for model-specific response parsing."""

    # Subclasses override these
    ASSISTANT_MARKERS = []
    EOS_TOKENS = []
    THINK_OPEN = "<think>"
    THINK_CLOSE = "</think>"
    ANSWER_OPEN = "<answer>"
    ANSWER_CLOSE = "</answer>"
    REQUIRE_ANSWER_TAGS = True

    def __init__(self, model_type: str):
        self.model_type = model_type

    def extract_assistant_response(self, full_text: str) -> Optional[str]:
        """Extract the assistant's response from full prompt+response text.

        Uses the last occurrence to handle double assistant turns
        (e.g. Qwen3's double <|im_start|>assistant pattern).
        """
        # Try "Assistant:" first (generic format used in some datasets)
        if "Assistant:" in full_text:
            return full_text.rsplit("Assistant:", 1)[1]

        for marker in self.ASSISTANT_MARKERS:
            if marker in full_text:
                return full_text.rsplit(marker, 1)[1]

        return None

    def extract_answer(self, processed_str: str) -> Optional[str]:
        """Extract the final answer from the assistant's response.

        If REQUIRE_ANSWER_TAGS is True, extracts from <answer>...</answer> tags.
        Otherwise, extracts text after </think>, optionally using <answer> tags if present.
        """
        if self.REQUIRE_ANSWER_TAGS:
            answer_pattern = r'<answer>(.*?)</answer>'
            matches = list(re.finditer(answer_pattern, processed_str, re.DOTALL))
            if not matches:
                print("[Error] No valid answer tags found")
                return None
            return matches[-1].group(1).strip()
        else:
            # Extract text after </think>
            if self.THINK_CLOSE in processed_str:
                answer_text = processed_str.split(self.THINK_CLOSE, 1)[1].strip()
                answer_text = self.strip_special_tokens(answer_text)
                # If <answer> tags are present, use them
                answer_match = re.search(r'<answer>(.*?)</answer>', answer_text, re.DOTALL)
                if answer_match:
                    answer_text = answer_match.group(1).strip()
                return answer_text if answer_text else None
            else:
                print("[Error] No </think> tag found")
                return None

    def validate_structure(self, processed_str: str) -> bool:
        """Check if the response has valid thinking+answer structure."""
        print("\n[Structure Validation]")
        validation_passed = True

        tags = {
            'think_start': (self.THINK_OPEN, 1),
            'think_end': (self.THINK_CLOSE, 1),
        }
        if self.REQUIRE_ANSWER_TAGS:
            tags['answer_start'] = (self.ANSWER_OPEN, 1)
            tags['answer_end'] = (self.ANSWER_CLOSE, 1)

        positions = {}
        for tag_name, (tag_str, expected_count) in tags.items():
            count = processed_str.count(tag_str)
            positions[tag_name] = pos = processed_str.find(tag_str)

            print(f"  {tag_str}: count={count}, position={pos}")

            if count != expected_count:
                print(f"  [Error] {tag_str} appears {count} times (expected {expected_count})")
                validation_passed = False

        if self.REQUIRE_ANSWER_TAGS:
            if (positions['think_start'] > positions['think_end'] or
                positions['think_end'] > positions['answer_start'] or
                positions['answer_start'] > positions['answer_end']):
                print("  [Error] Incorrect tag order: Expected <think>...</think><answer>...</answer>")
                validation_passed = False
            else:
                print("  Tag sequence validation passed")
        else:
            if positions['think_start'] > positions['think_end']:
                print("  [Error] Incorrect tag order: Expected <think>...</think>")
                validation_passed = False
            else:
                print("  Tag sequence validation passed")

        return validation_passed

    def strip_special_tokens(self, text: str) -> str:
        """Remove model-specific EOS/special tokens."""
        for token in self.EOS_TOKENS:
            text = text.replace(token, '').strip()
        return text

    def split_thinking(self, response: str) -> Tuple[str, str]:
        """Split response into (thinking_part, answer_part) around </think>.

        The thinking_part includes exactly one leading <think> and the </think>.
        The answer_part is everything after the first </think>.

        The opening <think> is added ONLY when the response does not already begin
        with one. Behavior/dialogue rollouts (rl_dataset builds the prompt with
        add_generation_prompt=True and no generation_prefix) contain their own
        leading <think>, so unconditionally prepending produced a malformed
        "<think><think>...</think>" that was then fed to the reward model. Direct-ToM
        rollouts (tom_dataset uses generation_prefix="<think>") start *after* the tag
        and still need it added. Checking the prefix handles both without doubling.
        """
        parts = response.split(self.THINK_CLOSE, 1)
        prefix = parts[0]
        if prefix.lstrip().startswith(self.THINK_OPEN):
            thinking = prefix + self.THINK_CLOSE
        else:
            thinking = self.THINK_OPEN + prefix + self.THINK_CLOSE
        answer = parts[1] if len(parts) > 1 else ""
        return thinking, answer

    def has_format_violation(self, response: str) -> bool:
        """True if the model's OWN response is malformed.

        A violation is: not exactly one <think> and one </think>, or an empty /
        malformed answer after </think>. Used as a graded format-reward penalty so
        dialogue-continuation training does not erode the <think>/answer structure
        the eval depends on. For REQUIRE_ANSWER_TAGS parsers a proper non-empty
        <answer>...</answer> is required; otherwise any non-empty content after
        </think> (with stray answer tags stripped) is accepted.
        """
        if response.count(self.THINK_OPEN) != 1 or response.count(self.THINK_CLOSE) != 1:
            return True
        _, answer_part = self.split_thinking(response)
        ans = self.strip_special_tokens(answer_part)
        if self.REQUIRE_ANSWER_TAGS:
            m = re.search(r'<answer>(.*?)</answer>', ans, re.DOTALL)
            return (m is None) or (len(m.group(1).strip()) == 0)
        ans_clean = re.sub(r'</?answer>', '', ans).strip()
        return len(ans_clean) == 0

    def build_stitched_response(self, thinking: str, ground_truth: str,
                                model_used_answer_tags: bool) -> str:
        """Build stitched response for actor-as-RM: thinking + ground truth answer.

        Args:
            thinking: The thinking part including <think>...</think> tags.
            ground_truth: The ground truth answer text.
            model_used_answer_tags: Whether the model's original response used <answer> tags.
        """
        if self.REQUIRE_ANSWER_TAGS or model_used_answer_tags:
            actual_response = self.ANSWER_OPEN + ground_truth + self.ANSWER_CLOSE
        else:
            actual_response = ground_truth
        return thinking + actual_response

    def answer_token_start(self, tokenizer, rendered_full: str,
                           thinking_length: int = None) -> int:
        """Return the index of the FIRST answer-region token in the tokenised
        ``rendered_full`` chat string (the token immediately after ``</think>``).

        Issue 6: the previous heuristic derived the boundary from re-tokenising the
        thinking-only chat (``apply_chat_template(chat + thinking, add_generation_prompt
        =False)``) and subtracting a fixed ``-1`` for the assistant-turn terminator. That
        terminator is model-dependent (1 token for Qwen's ``<|im_end|>`` but multiple for
        Gemma's ``<end_of_turn>\\n``), so a single ``-1`` under-covered the ground-truth
        answer on Gemma by more than one token. This method instead uses the tokenizer's
        character ``offset_mapping`` to find the exact first token whose span begins at or
        after the end of the last ``</think>`` — robust to any turn structure / terminator
        token count and to BPE merges around the boundary.

        Falls back to ``thinking_length`` (the caller's re-tokenised length) when a fast
        tokenizer / offset mapping is unavailable or no ``</think>`` is present.
        """
        close_idx = rendered_full.rfind(self.THINK_CLOSE)
        if close_idx == -1:
            return thinking_length
        answer_char_start = close_idx + len(self.THINK_CLOSE)
        try:
            enc = tokenizer(rendered_full, return_offsets_mapping=True,
                            add_special_tokens=False)
            offsets = enc['offset_mapping']
        except (TypeError, KeyError, NotImplementedError):
            return thinking_length
        if not offsets:
            return thinking_length
        for t, (s, e) in enumerate(offsets):
            # first token that contains at least one answer-region character
            if e > answer_char_start:
                return t
        return len(offsets)


class QwenResponseParser(ModelResponseParser):
    """Parser for Qwen2/Qwen2.5 models."""
    ASSISTANT_MARKERS = ["<|im_start|>assistant"]
    EOS_TOKENS = ["<|im_end|>", "<|endoftext|>"]
    REQUIRE_ANSWER_TAGS = True


class Qwen3ResponseParser(QwenResponseParser):
    """Parser for Qwen3 models — same markers, but answer tags optional."""
    REQUIRE_ANSWER_TAGS = False


class GemmaResponseParser(ModelResponseParser):
    """Parser for Gemma/Gemma2 models."""
    ASSISTANT_MARKERS = ["<start_of_turn>model"]
    EOS_TOKENS = ["<end_of_turn>", "<eos>"]
    REQUIRE_ANSWER_TAGS = False


_PARSERS = {
    'qwen2': QwenResponseParser,
    'qwen3': Qwen3ResponseParser,
    'gemma': GemmaResponseParser,
    'gemma2': GemmaResponseParser,
}


def get_parser(model_type: str) -> ModelResponseParser:
    """Factory function to get the right parser for a model type."""
    parser_cls = _PARSERS.get(model_type, QwenResponseParser)
    return parser_cls(model_type)
