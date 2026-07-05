#!/usr/bin/env python3
"""Converter for CraigslistBargain — stanfordnlp/craigslist_bargains."""

from scripts.dialogue_converter_base import DialogueConverterBase


class CraigslistBargainConverter(DialogueConverterBase):
    DATA_SOURCE = "craigslist_bargain"

    def download_dataset(self):
        from datasets import load_dataset

        ds = load_dataset("stanfordnlp/craigslist_bargains", split=self.split,
                          revision="refs/convert/parquet")
        conversations = []
        for idx, row in enumerate(ds):
            roles = row.get("agent_info", {}).get("Role", ["buyer", "seller"])
            turns = row.get("agent_turn", [])
            utts = row.get("utterance", [])
            dialog, speakers = [], []
            for i, text in enumerate(utts):
                text = (text or "").strip()
                if not text:
                    continue
                agent = turns[i] if i < len(turns) else i % 2
                dialog.append(text)
                speakers.append(roles[agent] if agent < len(roles) else f"agent_{agent}")
            if len(dialog) >= 2:
                conversations.append({
                    "dialog": dialog,
                    "speakers": speakers,
                    "conv_id": f"craigslist_{idx}",
                    "meta": {},
                })
        print(f"Loaded {len(conversations)} CraigslistBargain conversations")
        return conversations
