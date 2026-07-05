"""
Shared prompt templates for dialogue dataset converters.

This module consolidates all system prompt styles and user prompt templates
used across converters (DailyDialog, EmpatheticDialogues, etc.).
"""

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant helping with conversation analysis and generation."
)

RESEARCH_SYSTEM_PROMPT = (
    "You are an expert linguist and communication assistant helping with "
    "research on the psychology of interactions."
)

EMPATHY_SYSTEM_PROMPT = (
    "You are an empathetic conversational assistant skilled at understanding "
    "emotions and responding with care and sensitivity."
)

COT_SYSTEM_PROMPT = (
    "You are a helpful assistant. The assistant first thinks about the "
    "reasoning process in the mind and then provides the user with the answer. "
    "The reasoning process and answer are enclosed within <think> </think> and "
    "<answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>."
)

COT_TOM_SYSTEM_PROMPT = (
    "You are a helpful assistant skilled at understanding people's minds in conversations. "
    "Before answering, you must reason carefully inside <think> </think> tags. "
    "In your thinking, follow these steps:\n"
    "1. **Intents**: What is each party trying to achieve in this conversation?\n"
    "2. **Beliefs**: What does each party believe about the situation and about each other?\n"
    "3. **Goals**: What are each party's immediate and underlying goals?\n"
    "4. **Response strategy**: Given the above, what response would be most natural and appropriate?\n"
    "Then provide your answer inside <answer> </answer> tags, i.e., "
    "<think> reasoning here </think><answer> answer here </answer>."
)

COT_EVAL_SYSTEM_PROMPT = (
    "You are a helpful assistant. The assistant first thinks about the "
    "reasoning process in the mind and then provides the user with the answer. "
    "The reasoning process and answer are enclosed within <think> </think> and "
    "<answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>. "
    "Now the user asks you to solve a theory of mind reasoning problem. "
    "After thinking, when you finally reach a conclusion, clearly state "
    "your answer within <answer> </answer> tags."
)

COT_TOM2_SYSTEM_PROMPT = (
    "You are a helpful assistant. The assistant first thinks about the "
    "reasoning process in the mind and then provides the user with the answer. "
    "The reasoning process and answer are enclosed within <think> </think> and "
    "<answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>. "
    "In your reasoning, ensure you're thinking from all parties perspective "
    "before making your final reasoning."
)

# Tag-free CoT variant for models that reason in a native <think> mode and are
# NOT expected to emit <answer> tags (e.g. Qwen3, Gemma / Gemma2). It keeps the
# reason-then-answer + ToM framing of ``cot_eval`` but instructs NO <answer>
# tags, so it must be paired with ``add_response_tags: false`` and an empty
# ``generation_prefix``. See ``validate_prompt_tag_consistency``.
COT_EVAL_NOTAGS_SYSTEM_PROMPT = (
    "You are a helpful assistant. The assistant first thinks about the "
    "reasoning process in the mind and then provides the user with the answer. "
    "Now the user asks you to solve a theory of mind reasoning problem. "
    "Please reason step by step, and then clearly state your final answer."
)

SYSTEM_PROMPT_STYLES = {
    "default": DEFAULT_SYSTEM_PROMPT,
    "research": RESEARCH_SYSTEM_PROMPT,
    "empathy": EMPATHY_SYSTEM_PROMPT,
    "cot": COT_SYSTEM_PROMPT,
    "cot_tom": COT_TOM_SYSTEM_PROMPT,
    "cot_eval": COT_EVAL_SYSTEM_PROMPT,
    "cot_tom2": COT_TOM2_SYSTEM_PROMPT,
    "cot_eval_notags": COT_EVAL_NOTAGS_SYSTEM_PROMPT,
}

# ---------------------------------------------------------------------------
# User prompt templates
# ---------------------------------------------------------------------------

USER_TEMPLATE_SIMPLE = """Below is a real conversation between two people.
Based on the conversation history, predict what {responding_speaker} will say next.

Dialogue History:
{dialogue_history}

Now respond with what {responding_speaker} will say next without including the speaker: prefix."""


USER_TEMPLATE_HINT = """Below is a real conversation between two people.
Respond with the next utterance in the conversation as realistically as possible.

CONVERSATION:
{dialogue_history}

{responding_speaker}: """



PROMPT_STYLES = {
    "simple": USER_TEMPLATE_SIMPLE,
    "prefix_hint": USER_TEMPLATE_HINT,
}


# ---------------------------------------------------------------------------
# Tag-consistency validation
# ---------------------------------------------------------------------------

def system_prompt_instructs_answer_tags(system_prompt: str) -> bool:
    """Return True if *system_prompt* tells the model to emit <answer> tags."""
    return "<answer>" in (system_prompt or "")


def validate_prompt_tag_consistency(
    system_prompt: str,
    add_response_tags: bool,
    generation_prefix: str = "",
    context: str = "",
) -> list:
    """Check that the answer-tag signals agree across the three knobs.

    The three tag-related settings that must agree are:
      1. the system prompt (does it *instruct* <answer> tags?),
      2. ``add_response_tags`` (does the *target* get wrapped in <answer> tags?),
      3. ``generation_prefix`` (a "<think>" prefix implies a tagged CoT format).

    Returns a list of human-readable warning strings (empty if consistent).
    Raising is left to the caller so callers can choose warn-vs-error.
    """
    warnings = []
    prefix = f"[{context}] " if context else ""
    instructs_tags = system_prompt_instructs_answer_tags(system_prompt)

    if instructs_tags and not add_response_tags:
        warnings.append(
            f"{prefix}System prompt instructs <answer> tags but add_response_tags=False: "
            "the model will be told to emit tags the target lacks. Use a tag-free "
            "system prompt (e.g. system_prompt_style='cot_eval_notags') or set "
            "add_response_tags=True."
        )
    if add_response_tags and not instructs_tags:
        warnings.append(
            f"{prefix}add_response_tags=True but the system prompt does not mention "
            "<answer> tags: the target is wrapped in tags the prompt never asks for. "
            "Use a tag-instructing system prompt (e.g. 'cot_eval') or set "
            "add_response_tags=False."
        )
    if generation_prefix and "<think>" in generation_prefix and not instructs_tags:
        warnings.append(
            f"{prefix}generation_prefix contains '<think>' but the system prompt does "
            "not describe the <think>/<answer> tag format. For native-thinking models "
            "(Qwen3/Gemma) leave generation_prefix empty."
        )
    return warnings
