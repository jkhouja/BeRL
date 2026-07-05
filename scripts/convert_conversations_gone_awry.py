#!/usr/bin/env python3
"""
Converter for the ConvoKit Conversations Gone Awry corpus.

Downloads the corpus via convokit and converts multi-turn Wikipedia Talk page
conversations into the standard training parquet format.
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
)


class ConversationsGoneAwryConverter:
    """Converts ConvoKit conversations-gone-awry-corpus to training format."""

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
        data_source: str = "conversations_gone_awry",
        generation_prefix: str = "",
        include_system_in_prompt: bool = False,
        add_response_tags: bool = False,
        response_tag_open: str = "<answer>",
        response_tag_close: str = "</answer>",
        limit_turn: Optional[int] = None,
        multi_sample: bool = True,
        turn_order: str = "random",
        config: Optional[Dict[str, Any]] = None,
    ):
        if config:
            self.min_turns = config.get("min_turns", min_turns)
            self.max_turns = config.get("max_turns", max_turns)
            self.min_response_words = config.get("min_response_words", min_response_words)
            self.max_response_words = config.get("max_response_words", max_response_words)
            self.sample_size = config.get("sample_size", sample_size)
            self.seed = config.get("seed", seed)
            self.split = config.get("split", split)
            self.prompt_style = config.get("prompt_style", prompt_style)
            self.system_prompt_style = config.get("system_prompt_style", system_prompt_style)
            self.ability = config.get("ability", ability)
            self.data_source = config.get("data_source", data_source)
            self.generation_prefix = config.get("generation_prefix", generation_prefix)
            self.include_system_in_prompt = config.get("include_system_in_prompt", include_system_in_prompt)
            self.add_response_tags = config.get("add_response_tags", add_response_tags)
            self.response_tag_open = config.get("response_tag_open", response_tag_open)
            self.response_tag_close = config.get("response_tag_close", response_tag_close)
            self.limit_turn = config.get("limit_turn", limit_turn)
            self.multi_sample = config.get("multi_sample", multi_sample)
            self.turn_order = config.get("turn_order", turn_order)
            self.user_template = config.get("user_template", user_template)
            self.system_prompt = config.get("system_prompt", system_prompt)
        else:
            self.min_turns = min_turns
            self.max_turns = max_turns
            self.min_response_words = min_response_words
            self.max_response_words = max_response_words
            self.sample_size = sample_size
            self.seed = seed
            self.split = split
            self.prompt_style = prompt_style
            self.system_prompt_style = system_prompt_style
            self.ability = ability
            self.data_source = data_source
            self.generation_prefix = generation_prefix
            self.include_system_in_prompt = include_system_in_prompt
            self.add_response_tags = add_response_tags
            self.response_tag_open = response_tag_open
            self.response_tag_close = response_tag_close
            self.limit_turn = limit_turn
            self.multi_sample = multi_sample
            self.turn_order = turn_order
            self.user_template = user_template
            self.system_prompt = system_prompt

        if self.user_template is None:
            self.user_template = PROMPT_STYLES.get(self.prompt_style, USER_TEMPLATE_SIMPLE)
        if self.system_prompt is None:
            self.system_prompt = SYSTEM_PROMPT_STYLES.get(self.system_prompt_style, DEFAULT_SYSTEM_PROMPT)

        random.seed(self.seed)

    def download_dataset(self):
        """Download the corpus via convokit."""
        from convokit import Corpus, download

        print("Downloading conversations-gone-awry-corpus via convokit...")
        corpus = Corpus(filename=download("conversations-gone-awry-corpus"))
        print(f"Loaded corpus: {len(corpus.get_conversation_ids())} conversations, "
              f"{len(corpus.get_utterance_ids())} utterances")

        # Convert to list of conversation dicts
        conversations = []
        for conv in corpus.iter_conversations():
            meta = conv.meta
            conv_split = meta.get("split", "train")

            # Filter by requested split
            if conv_split != self.split:
                continue

            # Get utterances in order
            utts = conv.get_chronological_utterance_list()

            # Build dialogue: list of (speaker, text, metadata) tuples
            dialog = []
            for utt in utts:
                text = utt.text.strip()
                # Skip section headers (e.g., "==Topic==") — they're not real utterances
                if utt.meta.get("is_section_header", False):
                    continue
                if not text:
                    continue
                dialog.append({
                    "speaker": utt.speaker.id,
                    "text": text,
                    "toxicity": utt.meta.get("toxicity", 0.0),
                    "has_attack": utt.meta.get("comment_has_personal_attack", False),
                })

            if len(dialog) < 2:
                continue

            conversations.append({
                "dialog": [d["text"] for d in dialog],
                "speakers": [d["speaker"] for d in dialog],
                "toxicity": [d["toxicity"] for d in dialog],
                "has_attack": [d["has_attack"] for d in dialog],
                "conv_has_attack": meta.get("conversation_has_personal_attack", False),
                "page_title": meta.get("page_title", ""),
                "pair_id": meta.get("pair_id", ""),
                "conv_id": conv.id,
            })

        print(f"Loaded {len(conversations)} conversations for split='{self.split}'")
        attack_count = sum(1 for c in conversations if c["conv_has_attack"])
        print(f"  Attack: {attack_count}, Civil: {len(conversations) - attack_count}")
        return conversations

    def filter_conversation(self, dialogue: List[str]) -> bool:
        """Check if a conversation meets filtering criteria."""
        if len(dialogue) < self.min_turns:
            return False
        if self.max_turns and len(dialogue) > self.max_turns:
            return False
        for utterance in dialogue:
            word_count = len(utterance.split())
            if word_count < self.min_response_words:
                return False
            if self.max_response_words and word_count > self.max_response_words:
                return False
        return True

    def create_training_examples(
        self,
        item: Dict[str, Any],
        conv_idx: int,
    ) -> List[Dict[str, Any]]:
        """Create training examples from a conversation."""
        dialogue = item["dialog"]
        speakers = item["speakers"]
        examples = []

        for split_idx in range(1, len(dialogue)):
            if self.limit_turn is not None and split_idx < self.limit_turn:
                continue

            # Build dialogue history using actual speaker names
            dialogue_hist = []
            for i in range(split_idx):
                dialogue_hist.append(f"{speakers[i]}: {dialogue[i]}")

            response = dialogue[split_idx]
            response_words = len(response.split())
            responding_speaker = speakers[split_idx]

            dialogue_history_str = "\n".join(dialogue_hist)

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
                "reward_model": {
                    "ground_truth": response,
                    "style": "rule",
                },
                "metadata": {
                    "conv_id": item["conv_id"],
                    "turn": split_idx,
                    "total_turns": len(dialogue),
                    "responding_speaker": responding_speaker,
                    "dialogue_history": dialogue_history_str,
                    "conv_has_attack": item["conv_has_attack"],
                    "page_title": item["page_title"],
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
        """Convert conversations to training DataFrame."""
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

        df = pd.DataFrame(all_examples)
        return df
