#!/usr/bin/env python3
"""
Shared base class for *narrative* (non-conversational) dataset converters.

BeRL's behaviour-prediction reward is dataset-agnostic: it rewards the
log-likelihood of the *real* next span given [context + CoT]. The conversational
converters (``DialogueConverterBase``) predict the next *utterance* from a
dialogue history. This base extends the same machinery to predict the next
*sentence / line* from a narrative context (stories, scripts, plays), so the
identical ll_loss reward path, MC/optctx builders and ``build_dataset.py``
pipeline all work unchanged.

A narrative converter subclasses this and implements ``download_dataset()``
returning a list of story dicts::

    {"dialog": [str, ...],      # ordered sentences / lines of the narrative
     "speakers": [str, ...] | None,  # optional per-line speaker (scripts/plays);
                                     # None => pure narration
     "conv_id": str,
     "meta": {"premise": str | None, ...}}  # optional premise (e.g. a writing
                                            # prompt) shown before the context

Differences vs ``DialogueConverterBase``:
  * Context lines are joined as prose (optionally prefixed by ``speaker:`` only
    when speakers are provided), NOT always ``speaker: text``.
  * An optional ``premise`` (writing prompt / title) is shown before the context.
  * A narrative user template ("predict the next sentence/line") is used.
  * Word-count bounds are enforced on the *target* span only (long earlier
    context never rejects the whole story), and an optional sliding
    ``max_context_sentences`` window caps very long prefixes.

Every emitted row has the SAME schema as the dialogue converters
(``reward_model.ground_truth`` = the real next span, ``prompt`` / ``raw_prompt``,
tags, ``generation_prefix``, metadata), so it is a drop-in behaviour source.
"""

from typing import Any, Dict, List, Optional

from scripts.dialogue_converter_base import DialogueConverterBase
from scripts.prompt_templates import (
    PROMPT_STYLES,
    USER_TEMPLATE_NARRATIVE,
)

import re

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])[\"')\]]*\s+")


def normalize_text(text: str) -> str:
    """Clean space-padded punctuation / PTB tokenisation common in screenplay
    and EMNLP-tokenised corpora, WITHOUT splitting into sentences.

    Fixes: literal ``<newline>`` markup, PTB quote tokens (`` `` ``/``''`` ->
    straight quotes, tightened directionally), spaced clitics (``do n't`` ->
    ``don't``), and space-before-punctuation (``foo .`` -> ``foo.``).
    """
    if not text:
        return ""
    # strip literal newline/markup tokens common in WritingPrompts dumps
    text = re.sub(r"<\s*/?\s*newline\s*>", " ", text, flags=re.IGNORECASE)
    text = text.replace("\n", " ")
    # directional quote tightening on PTB tokens ( `` foo '' -> "foo" )
    text = re.sub(r"``\s*", '"', text)
    text = re.sub(r"\s*''", '"', text)
    text = re.sub(r"\s+", " ", text).strip()
    # tighten clitics ( she 's -> she's , do n't -> don't )
    text = re.sub(r"\s+n't\b", "n't", text)
    text = re.sub(r"\s+'\s*(s|t|re|ve|ll|d|m)\b", r"'\1", text)
    # common PTB two-token contractions
    text = re.sub(r"\bgon na\b", "gonna", text, flags=re.IGNORECASE)
    text = re.sub(r"\bwan na\b", "wanna", text, flags=re.IGNORECASE)
    text = re.sub(r"\bgot ta\b", "gotta", text, flags=re.IGNORECASE)
    # normalise space-before-punctuation
    text = re.sub(r"\s+([.!?,;:])", r"\1", text)
    return text


def split_sentences(text: str) -> List[str]:
    """Lightweight sentence splitter (no nltk dependency).

    Handles both normal prose and the space-padded punctuation common in
    WritingPrompts / EMNLP-tokenised corpora (e.g. ``foo . Bar``). Collapses
    internal whitespace and drops empty fragments.
    """
    if not text:
        return []
    text = normalize_text(text)
    parts = _SENT_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p and p.strip()]


class NarrativeConverterBase(DialogueConverterBase):
    """Base converter for narrative next-sentence / next-line prediction."""

    DATA_SOURCE = "narrative"

    def __init__(self, *args, **kwargs):
        # Default to the narrative prompt style + ability unless a config overrides.
        cfg = (kwargs.get("config") or {})
        cfg.setdefault("prompt_style", "narrative")
        cfg.setdefault("ability", "narrative_generation")
        kwargs["config"] = cfg
        super().__init__(*args, **kwargs)
        # Retain the resolved config so subclasses can read source-specific knobs.
        self.config = cfg
        # If the resolved user_template is still the dialogue default (because
        # the requested prompt_style wasn't found), force the narrative template.
        if self.prompt_style == "narrative" and "{dialogue_history}" not in (self.user_template or ""):
            self.user_template = USER_TEMPLATE_NARRATIVE
        # Narrative-specific knobs (read straight from config; safe defaults).
        self.min_context_sentences = int(cfg.get("min_context_sentences", 2))
        self.max_context_sentences = cfg.get("max_context_sentences", None)
        self.premise_label = cfg.get("premise_label", "Writing prompt")
        self.narrator_label = cfg.get("narrator_label", "the narrator")

    # ------------------------------------------------------------------
    # Filtering: only require a minimum number of lines; the per-target
    # word-count bounds are enforced in create_training_examples so a long
    # earlier sentence never rejects an otherwise good story.
    # ------------------------------------------------------------------
    def filter_conversation(self, dialogue: List[str]) -> bool:
        return len(dialogue) >= max(self.min_turns, self.min_context_sentences + 1)

    def _format_context(self, lines: List[str], speakers: Optional[List[str]],
                        start: int, end: int, premise: Optional[str]) -> str:
        """Build the narrative context string for target index ``end`` from
        lines[start:end]. Sliding window keeps at most max_context_sentences."""
        lo = start
        if self.max_context_sentences:
            lo = max(start, end - int(self.max_context_sentences))
        if speakers is not None:
            body = "\n".join(f"{speakers[i]}: {lines[i]}" for i in range(lo, end))
        else:
            # pure narration: join as prose
            body = " ".join(lines[i].strip() for i in range(lo, end))
        if premise:
            return f"{self.premise_label}: {premise.strip()}\n\n{body}"
        return body

    def create_training_examples(self, item: Dict[str, Any], conv_idx: int) -> List[Dict[str, Any]]:
        lines = item["dialog"]
        speakers = item.get("speakers")
        premise = (item.get("meta", {}) or {}).get("premise")
        extra_meta = {k: v for k, v in (item.get("meta", {}) or {}).items() if k != "premise"}
        examples: List[Dict[str, Any]] = []

        for tgt in range(self.min_context_sentences, len(lines)):
            if self.limit_turn is not None and tgt < self.limit_turn:
                continue
            response = (lines[tgt] or "").strip()
            response_words = len(response.split())
            if response_words < self.min_response_words:
                continue
            if self.max_response_words and response_words > self.max_response_words:
                continue

            responding_speaker = (speakers[tgt] if speakers is not None else self.narrator_label)
            context_str = self._format_context(lines, speakers, 0, tgt, premise)

            user_prompt = self.user_template.format(
                dialogue_history=context_str,
                responding_speaker=responding_speaker,
            )

            prompt = [{"content": user_prompt, "role": "user"}]
            if self.include_system_in_prompt:
                prompt.insert(0, {"content": self.system_prompt, "role": "system"})

            raw_prompt = [
                {"content": self.system_prompt, "role": "system"},
                {"content": user_prompt, "role": "user"},
            ]

            response_with_tags = response
            if self.add_response_tags:
                response_with_tags = f"{self.response_tag_open}{response}{self.response_tag_close}"

            example = {
                "prompt_len": None,
                "response_words": response_words,
                "data_source": self.data_source,
                "prompt": prompt,
                "raw_system_prompt": self.system_prompt,
                "raw_user_prompt": user_prompt,
                "prompt_for_pp": None,
                "ability": self.ability,
                "reward_model": {"ground_truth": response, "style": "rule"},
                "metadata": {
                    "conv_id": item["conv_id"],
                    "turn": tgt,
                    "total_turns": len(lines),
                    "responding_speaker": responding_speaker,
                    "dialogue_history": context_str,
                    **extra_meta,
                },
                "answer_pp": None,
                "raw_prompt": raw_prompt,
            }
            if self.generation_prefix:
                example["generation_prefix"] = self.generation_prefix
            if self.add_response_tags:
                example["response_with_tags"] = response_with_tags

            examples.append(example)

        return examples
