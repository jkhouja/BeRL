#!/usr/bin/env python3
"""Converter for MovieSum — rohitsaxena/MovieSum.

2,200 movie screenplays (1800/200/200) shipped as XML-tagged ``script`` strings
with an explicit tag vocabulary::

    <scene> <stage_direction> <scene_description>
            <character> <dialogue> <parenthetical> ... </scene>

Unlike the prose narrative sources (ROCStories / WritingPrompts / TinyStories),
MovieSum has **named speakers**, so we build genuine multi-party dialogue: each
``<scene>`` becomes one coherent conversation and the model predicts the *next
character's line* given the prior turns in that scene (a real Theory-of-Mind
signal — character intent — at narrative scale, mirroring the conversational
anchor but on movie scripts).

Design (scene-level, "script" prompt style, speakers = character names):
  * Split each script on ``<scene>`` boundaries → many bounded exchanges per
    film (scene-coherent context, no giant 900-turn sequences).
  * Within a scene, extract the ordered ``<character>``/``<dialogue>`` pairs.
  * The scene heading (``<stage_direction>``) + optional first
    ``<scene_description>`` become the ``premise`` (setting grounding).
  * ``NarrativeConverterBase`` then renders context as ``SPEAKER: line`` turns
    and emits one target per character line from ``min_context_sentences`` on,
    with the identical row schema / behaviour (ll_loss) reward path.
"""

import random
import re

from scripts.narrative_converter_base import NarrativeConverterBase, normalize_text

_SCENE_RE = re.compile(r"<scene>(.*?)</scene>", re.S)
_STAGE_RE = re.compile(r"<stage_direction>(.*?)</stage_direction>", re.S)
_DESC_RE = re.compile(r"<scene_description>(.*?)</scene_description>", re.S)
# character / dialogue tags in document order
_TURN_RE = re.compile(r"<(character|dialogue)>(.*?)</\1>", re.S)


class MovieSumConverter(NarrativeConverterBase):
    DATA_SOURCE = "moviesum"

    def download_dataset(self):
        from datasets import load_dataset

        include_desc = bool(self.config.get("include_scene_description", True))
        max_premise_words = int(self.config.get("max_premise_words", 60))
        want = (self.sample_size or 3000)
        # Each qualifying scene yields several targets, so we need far fewer
        # scenes than the final example count. Cap scene collection to bound
        # memory / build time; shuffle movie order (seed) to avoid alphabetical
        # bias from the natural dataset ordering.
        scene_cap = max(want * 2, 4000)

        ds = load_dataset("rohitsaxena/MovieSum", split=self.split)
        movie_order = list(range(len(ds)))
        random.Random(self.seed).shuffle(movie_order)

        stories = []
        for midx in movie_order:
            if len(stories) >= scene_cap:
                break
            row = ds[midx]
            script = row.get("script") or ""
            movie = (row.get("movie_name") or f"movie_{midx}").strip()
            for sidx, scene in enumerate(_SCENE_RE.findall(script)):
                if len(stories) >= scene_cap:
                    break
                # ordered character/dialogue pairs within the scene
                speakers, lines = [], []
                pending_char = None
                for tag, content in _TURN_RE.findall(scene):
                    txt = normalize_text(content)
                    if tag == "character":
                        pending_char = txt or None
                    elif tag == "dialogue" and txt:
                        speakers.append(pending_char or "UNKNOWN")
                        lines.append(txt)
                        pending_char = None
                if len(lines) < max(self.min_turns, self.min_context_sentences + 1):
                    continue
                # premise = scene heading (+ optional first action description)
                heading = ""
                m = _STAGE_RE.search(scene)
                if m:
                    heading = normalize_text(m.group(1))
                if include_desc:
                    d = _DESC_RE.search(scene)
                    if d:
                        desc = normalize_text(d.group(1))
                        heading = (heading + " " + desc).strip() if heading else desc
                if heading:
                    heading = " ".join(heading.split()[:max_premise_words])
                stories.append({
                    "dialog": lines,
                    "speakers": speakers,
                    "conv_id": f"moviesum_{midx}_{sidx}",
                    "meta": {"premise": heading or None, "movie": movie},
                })
        print(f"Loaded {len(stories)} MovieSum scenes (scene_cap={scene_cap})")
        return stories
