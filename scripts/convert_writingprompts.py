#!/usr/bin/env python3
"""Converter for WritingPrompts — vkpriya/GPT-WritingPrompts (human stories).

The repo ships ``human_wp_stories.json`` as a dict ``{prompt_text: [story, ...]}``
(97,223 prompts, human-written stories). We use the writing prompt as the
narrative premise and predict a later sentence of the story from the earlier
ones — a high-variance, creative long-form narrative source. Stories can be
very long, so a sliding ``max_context_sentences`` window (set in the dcfg) keeps
prompts bounded.
"""

import json

from scripts.narrative_converter_base import NarrativeConverterBase, split_sentences


class WritingPromptsConverter(NarrativeConverterBase):
    DATA_SOURCE = "writingprompts"

    def download_dataset(self):
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            "vkpriya/GPT-WritingPrompts", "human_wp_stories.json", repo_type="dataset"
        )
        with open(path) as f:
            data = json.load(f)

        want = (self.sample_size or 2000)
        cap = max(want * 4, 4000)
        stories = []
        for pidx, (prompt_text, story_list) in enumerate(data.items()):
            if len(stories) >= cap:
                break
            if not isinstance(story_list, list):
                story_list = [story_list]
            for sidx, story in enumerate(story_list):
                if len(stories) >= cap:
                    break
                sents = split_sentences(story or "")
                if len(sents) < 3:
                    continue
                premise = (prompt_text or "").strip()
                # Strip literal newline markup + a common "[ WP ]" tag prefix.
                import re as _re
                premise = _re.sub(r"<\s*/?\s*newline\s*>", " ", premise, flags=_re.IGNORECASE)
                premise = _re.sub(r"\s+", " ", premise).strip()
                premise = premise.lstrip("[").replace("WP ]", "", 1).strip(" []")
                stories.append({
                    "dialog": sents,
                    "speakers": None,
                    "conv_id": f"writingprompts_{pidx}_{sidx}",
                    "meta": {"premise": premise or None},
                })
        print(f"Loaded {len(stories)} WritingPrompts stories (cap={cap})")
        return stories
