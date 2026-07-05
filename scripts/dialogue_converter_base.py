#!/usr/bin/env python3
"""
Shared base class for dialogue dataset converters.

Encapsulates the standard BeRL training parquet conversion: filtering,
turn-by-turn example creation (predict turn i+1 from history), sampling and
turn ordering. New converters subclass this and only implement
``download_dataset()``, returning a list of conversation dicts of the form::

    {"dialog": [str, ...], "speakers": [str, ...], "conv_id": str, "meta": {...}}

This mirrors the schema used by ``convert_conversations_gone_awry.py`` and
keeps all converters consistent with the pipeline in ``build_dataset.py``.
"""

import random
from typing import Optional, List, Dict, Any

import pandas as pd
from tqdm import tqdm

from scripts.prompt_templates import (
    SYSTEM_PROMPT_STYLES,
    PROMPT_STYLES,
    DEFAULT_SYSTEM_PROMPT,
    USER_TEMPLATE_SIMPLE,
    validate_prompt_tag_consistency,
)


class DialogueConverterBase:
    """Base converter sharing filtering and example-building logic."""

    # Subclasses set their default data_source identifier
    DATA_SOURCE = "dialogue"

    def __init__(
        self,
        min_turns: int = 4,
        max_turns: Optional[int] = None,
        min_response_words: int = 5,
        max_response_words: Optional[int] = None,
        sample_size: Optional[int] = None,
        seed: int = 42,
        split: str = "train",
        system_prompt: Optional[str] = None,
        user_template: Optional[str] = None,
        prompt_style: str = "simple",
        system_prompt_style: str = "default",
        ability: str = "conversation_generation",
        data_source: Optional[str] = None,
        generation_prefix: str = "",
        include_system_in_prompt: bool = False,
        add_response_tags: bool = False,
        response_tag_open: str = "<answer>",
        response_tag_close: str = "</answer>",
        limit_turn: Optional[int] = None,
        multi_sample: bool = True,
        turn_order: str = "random",
        target_speaker: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        cfg = config or {}
        self.min_turns = cfg.get("min_turns", min_turns)
        self.max_turns = cfg.get("max_turns", max_turns)
        self.min_response_words = cfg.get("min_response_words", min_response_words)
        self.max_response_words = cfg.get("max_response_words", max_response_words)
        self.sample_size = cfg.get("sample_size", sample_size)
        self.seed = cfg.get("seed", seed)
        self.split = cfg.get("split", split)
        self.prompt_style = cfg.get("prompt_style", prompt_style)
        self.system_prompt_style = cfg.get("system_prompt_style", system_prompt_style)
        self.ability = cfg.get("ability", ability)
        self.data_source = cfg.get("data_source", data_source or self.DATA_SOURCE)
        self.generation_prefix = cfg.get("generation_prefix", generation_prefix)
        self.include_system_in_prompt = cfg.get("include_system_in_prompt", include_system_in_prompt)
        self.add_response_tags = cfg.get("add_response_tags", add_response_tags)
        self.response_tag_open = cfg.get("response_tag_open", response_tag_open)
        self.response_tag_close = cfg.get("response_tag_close", response_tag_close)
        self.limit_turn = cfg.get("limit_turn", limit_turn)
        self.multi_sample = cfg.get("multi_sample", multi_sample)
        self.turn_order = cfg.get("turn_order", turn_order)
        # Optional: only build prediction examples for this speaker's turns
        # (e.g. "user"). When set, word-count limits are enforced only on the
        # target turn, so long context turns from other speakers don't reject
        # the whole conversation.
        self.target_speaker = cfg.get("target_speaker", target_speaker)
        self.user_template = cfg.get("user_template", user_template)
        self.system_prompt = cfg.get("system_prompt", system_prompt)

        if self.user_template is None:
            self.user_template = PROMPT_STYLES.get(self.prompt_style, USER_TEMPLATE_SIMPLE)
        if self.system_prompt is None:
            self.system_prompt = SYSTEM_PROMPT_STYLES.get(self.system_prompt_style, DEFAULT_SYSTEM_PROMPT)

        # Guard: the <answer>-tag signals (system prompt, add_response_tags,
        # generation_prefix) must agree. Mismatches silently hurt the reward
        # (e.g. a tag-free target with a system prompt that demands tags).
        for _w in validate_prompt_tag_consistency(
            system_prompt=self.system_prompt,
            add_response_tags=self.add_response_tags,
            generation_prefix=self.generation_prefix,
            context=f"{self.data_source} converter",
        ):
            print(f"[WARN] {_w}")

        random.seed(self.seed)

    def download_dataset(self):
        """Subclasses must return a list of conversation dicts:
        {"dialog": [str...], "speakers": [str...], "conv_id": str, "meta": {...}}"""
        raise NotImplementedError

    def filter_conversation(self, dialogue: List[str]) -> bool:
        if len(dialogue) < self.min_turns:
            return False
        if self.max_turns and len(dialogue) > self.max_turns:
            return False
        # When targeting a specific speaker, word-count limits are enforced
        # per-target-turn in create_training_examples instead (long context
        # turns from other speakers must not reject the whole conversation).
        if self.target_speaker is not None:
            return True
        for utterance in dialogue:
            word_count = len(utterance.split())
            if word_count < self.min_response_words:
                return False
            if self.max_response_words and word_count > self.max_response_words:
                return False
        return True

    def create_training_examples(self, item: Dict[str, Any], conv_idx: int) -> List[Dict[str, Any]]:
        dialogue = item["dialog"]
        speakers = item.get("speakers") or [f"Speaker{i % 2 + 1}" for i in range(len(dialogue))]
        extra_meta = item.get("meta", {})
        examples = []

        for split_idx in range(1, len(dialogue)):
            if self.limit_turn is not None and split_idx < self.limit_turn:
                continue

            dialogue_hist = [f"{speakers[i]}: {dialogue[i]}" for i in range(split_idx)]
            response = dialogue[split_idx]
            response_words = len(response.split())
            responding_speaker = speakers[split_idx]
            dialogue_history_str = "\n".join(dialogue_hist)

            # Optionally restrict prediction targets to a single speaker and
            # enforce word-count bounds on the target turn only.
            if self.target_speaker is not None:
                if responding_speaker != self.target_speaker:
                    continue
                if response_words < self.min_response_words:
                    continue
                if self.max_response_words and response_words > self.max_response_words:
                    continue

            user_prompt = self.user_template.format(
                dialogue_history=dialogue_history_str,
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
                    "turn": split_idx,
                    "total_turns": len(dialogue),
                    "responding_speaker": responding_speaker,
                    "dialogue_history": dialogue_history_str,
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

    def convert(self, conversations) -> pd.DataFrame:
        all_examples = []
        filtered_count = 0
        conv_count = 0

        print("Converting conversations to training examples...")
        for item in tqdm(conversations):
            if not self.filter_conversation(item["dialog"]):
                filtered_count += 1
                continue

            examples = self.create_training_examples(item, conv_count)
            if not self.multi_sample and examples:
                examples = [random.choice(examples)]

            all_examples.extend(examples)
            conv_count += 1

        print(f"Filtered out {filtered_count} conversations")
        print(f"Created {len(all_examples)} training examples from {conv_count} conversations")

        if self.sample_size and len(all_examples) > self.sample_size:
            print(f"Sampling {self.sample_size} examples from {len(all_examples)}")
            all_examples = random.sample(all_examples, self.sample_size)

        if self.turn_order == "early_first":
            all_examples.sort(key=lambda x: x["metadata"]["turn"])
        elif self.turn_order == "late_first":
            all_examples.sort(key=lambda x: -x["metadata"]["turn"])
        elif self.turn_order == "random":
            random.shuffle(all_examples)
        else:
            raise ValueError(f"Unknown turn_order: {self.turn_order}")

        return pd.DataFrame(all_examples)
