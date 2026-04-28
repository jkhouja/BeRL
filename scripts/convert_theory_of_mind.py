#!/usr/bin/env python3
"""
Converter for Theory of Mind training data.

Reads pre-built ToM parquet (hi_tom + explore_tom) and reshapes it
to match the pipeline's expected format for GRPO training.
"""

import random
import re
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


class TheoryOfMindConverter:
    """Converts cleaned ToM parquet data to the pipeline training format."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        cfg = config or kwargs
        self.sample_size = cfg.get("sample_size", None)
        self.data_source_filter = cfg.get("data_source_filter", None)  # 'hi_tom', 'explore_tom', or None for both
        self.seed = cfg.get("seed", 42)
        self.parquet_path = cfg.get(
            "parquet_path", "data/cleaned_tom/ToM_train_HiEx_hint.parquet"
        )
        self.generation_prefix = cfg.get("generation_prefix", "<think>")
        random.seed(self.seed)

    def download_dataset(self) -> pd.DataFrame:
        """Read the pre-built ToM parquet file."""
        print(f"Reading ToM data from: {self.parquet_path}")
        df = pd.read_parquet(self.parquet_path)
        print(f"  Loaded {len(df)} rows")

        if self.data_source_filter:
            df = df[df["data_source"] == self.data_source_filter].reset_index(drop=True)
            print(f"  Filtered to data_source='{self.data_source_filter}': {len(df)} rows")

        if self.sample_size and self.sample_size < len(df):
            df = df.sample(n=self.sample_size, random_state=self.seed).reset_index(drop=True)
            print(f"  Sampled {self.sample_size} rows")

        return df

    @staticmethod
    def _parse_prompt_content(content: str):
        """Parse the embedded system+user prompt from the ToM data.

        The existing ToM parquet stores everything in a single 'user' role message
        with <|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>
        formatting baked into the content string. We need to extract the actual
        system and user text.
        """
        # Extract system prompt
        sys_match = re.search(
            r"<\|im_start\|>system\n(.*?)<\|im_end\|>", content, re.DOTALL
        )
        # Extract user prompt (up to the assistant/think prefix)
        user_match = re.search(
            r"<\|im_start\|>user\n(.*?)<\|im_end\|>", content, re.DOTALL
        )

        system_text = sys_match.group(1).strip() if sys_match else ""
        user_text = user_match.group(1).strip() if user_match else content

        return system_text, user_text

    def convert(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform ToM dataframe to pipeline format."""
        rows = []
        for _, row in df.iterrows():
            # Parse the embedded prompt
            prompt_array = row["prompt"]
            # Convert numpy array to list if needed
            if isinstance(prompt_array, np.ndarray):
                prompt_array = prompt_array.tolist()

            content = prompt_array[0]["content"]
            system_text, user_text = self._parse_prompt_content(content)

            # Build raw_prompt: list of [system, user] dicts
            raw_prompt = [
                {"role": "system", "content": system_text},
                {"role": "user", "content": user_text},
            ]

            # Build prompt (what gets passed to chat template): same as raw_prompt
            prompt = np.array(raw_prompt, dtype=object)

            answer = row["answer"]
            reward_model = row["reward_model"]
            if isinstance(reward_model, np.ndarray):
                reward_model = reward_model.item()

            response_with_tags = f"<answer>{answer}</answer>"

            rows.append(
                {
                    "prompt_len": None,  # computed by pipeline
                    "response_words": len(answer.split()),
                    "data_source": row["data_source"],
                    "prompt": prompt,
                    "raw_system_prompt": system_text,
                    "raw_user_prompt": user_text,
                    "prompt_for_pp": None,
                    "ability": row["ability"],
                    "reward_model": reward_model,
                    "metadata": {
                        "story": row.get("story", ""),
                        "question": row.get("question", ""),
                    },
                    "answer_pp": None,  # computed by perplexity step
                    "raw_prompt": np.array(raw_prompt, dtype=object),
                    "generation_prefix": self.generation_prefix,
                    "response_with_tags": response_with_tags,
                }
            )

        result = pd.DataFrame(rows)
        print(f"  Converted {len(result)} rows to pipeline format")
        print(f"  Data sources: {result['data_source'].value_counts().to_dict()}")
        return result
