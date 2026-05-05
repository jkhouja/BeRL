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

SYSTEM_PROMPT_STYLES = {
    "default": DEFAULT_SYSTEM_PROMPT,
    "research": RESEARCH_SYSTEM_PROMPT,
    "empathy": EMPATHY_SYSTEM_PROMPT,
    "cot": COT_SYSTEM_PROMPT,
    "cot_tom": COT_TOM_SYSTEM_PROMPT,
    "cot_eval": COT_EVAL_SYSTEM_PROMPT,
}

# ---------------------------------------------------------------------------
# User prompt templates
# ---------------------------------------------------------------------------

SIMPLE_USER_TEMPLATE = """Below is a real conversation between two people.
Based on the conversation history, predict what {responding_speaker} will say next.

Dialogue History:
{dialogue_history}

Now respond with what {responding_speaker} will say next."""

DETAILED_USER_TEMPLATE = """Below is a real conversation between two people. Continue the conversation as realistically as possible.

CONVERSATION:
{dialogue_history}

Now respond with the following:
{responding_speaker}: """

BASELINE_USER_TEMPLATE = """Below is a real conversation between two people.
Respond with the next utterance in the conversation as realistically as possible.

CONVERSATION:
{dialogue_history}

{responding_speaker}: """

EMPATHY_USER_TEMPLATE = """Below is a conversation where one person is sharing an emotional experience and another is responding with empathy.

Context: {emotion_label}

CONVERSATION:
{dialogue_history}

Continue the conversation with an empathetic response as {responding_speaker}:
"""

COT_USER_TEMPLATE = """Below is a real conversation between two people.
Based on the conversation history, think about what {responding_speaker} would say next. Consider the context, tone, and flow of the conversation.

Dialogue History:
{dialogue_history}

Think step by step about what {responding_speaker} will say next, then provide the response."""

COT_TOM_USER_TEMPLATE = """Below is a real conversation between two people.
Based on the conversation history, figure out what {responding_speaker} would say next.

Dialogue History:
{dialogue_history}

Reason about each person's intent, beliefs, and goals, then predict what {responding_speaker} will say next."""

PROMPT_STYLES = {
    "simple": SIMPLE_USER_TEMPLATE,
    "detailed": DETAILED_USER_TEMPLATE,
    "baseline": BASELINE_USER_TEMPLATE,
    "empathy": EMPATHY_USER_TEMPLATE,
    "cot": COT_USER_TEMPLATE,
    "cot_tom": COT_TOM_USER_TEMPLATE,
}
