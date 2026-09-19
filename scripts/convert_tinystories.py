#!/usr/bin/env python3
"""Converter for TinyStories — roneneldan/TinyStories.

~2.1M short synthetic stories (simple vocabulary) as a single ``text`` field.
We sentence-split each story and predict a later sentence from the earlier ones.
Simple, clean narration — a low-complexity narrative point to contrast with the
richer ROCStories / WritingPrompts sources.
"""

from scripts.narrative_converter_base import NarrativeConverterBase, split_sentences


class TinyStoriesConverter(NarrativeConverterBase):
    DATA_SOURCE = "tinystories"

    def download_dataset(self):
        from datasets import load_dataset

        # Stream to avoid materialising the full 2.1M-row dataset; take enough
        # candidate stories to satisfy sample_size after filtering.
        want = (self.sample_size or 2000)
        cap = max(want * 6, 5000)
        ds = load_dataset("roneneldan/TinyStories", split=self.split, streaming=True)
        stories = []
        for idx, row in enumerate(ds):
            if len(stories) >= cap:
                break
            sents = split_sentences(row.get("text") or "")
            if len(sents) < 3:
                continue
            stories.append({
                "dialog": sents,
                "speakers": None,
                "conv_id": f"tinystories_{idx}",
                "meta": {"premise": None},
            })
        print(f"Loaded {len(stories)} TinyStories stories (streamed cap={cap})")
        return stories
