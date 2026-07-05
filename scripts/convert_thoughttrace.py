#!/usr/bin/env python3
"""Converter for ThoughtTrace — SCAI-JHU/ThoughtTrace.

Human–AI conversations where messages[].type is the role and messages[].reasons
captures the user's gold thought. Utterances are long (~231 words), so cap with
max_response_words in the pipeline config.
"""

from scripts.dialogue_converter_base import DialogueConverterBase


class ThoughtTraceConverter(DialogueConverterBase):
    DATA_SOURCE = "thoughttrace"

    def download_dataset(self):
        from datasets import load_dataset

        ds = load_dataset("SCAI-JHU/ThoughtTrace", split=self.split)
        conversations = []
        for idx, row in enumerate(ds):
            dialog, speakers = [], []
            for msg in row.get("messages", []):
                text = (msg.get("content") or "").strip()
                if not text:
                    continue
                role = msg.get("type", "user")
                dialog.append(text)
                speakers.append("user" if role in ("user", "human") else "assistant")
            if len(dialog) >= 2:
                conversations.append({
                    "dialog": dialog,
                    "speakers": speakers,
                    "conv_id": str(row.get("id", f"thoughttrace_{idx}")),
                    "meta": {},
                })
        print(f"Loaded {len(conversations)} ThoughtTrace conversations")
        return conversations
