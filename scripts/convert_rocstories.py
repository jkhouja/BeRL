#!/usr/bin/env python3
"""Converter for ROCStories — shawon/rocstories-combined.

98,161 five-sentence commonsense stories. Each ``sentences`` field is an ordered
list of exactly 5 sentences. We predict a later sentence from the earlier ones
(behaviour-prediction over narrative causal/character structure). ``title`` is
used as an optional premise so the model knows the story's topic.
"""

from scripts.narrative_converter_base import NarrativeConverterBase


class ROCStoriesConverter(NarrativeConverterBase):
    DATA_SOURCE = "rocstories"

    def download_dataset(self):
        from datasets import load_dataset

        ds = load_dataset("shawon/rocstories-combined", split=self.split)
        stories = []
        for idx, row in enumerate(ds):
            sents = [s.strip() for s in (row.get("sentences") or []) if s and s.strip()]
            if len(sents) < 3:
                continue
            title = (row.get("title") or "").strip()
            stories.append({
                "dialog": sents,
                "speakers": None,
                "conv_id": f"rocstories_{idx}",
                "meta": {"premise": title or None},
            })
        print(f"Loaded {len(stories)} ROCStories stories")
        return stories
