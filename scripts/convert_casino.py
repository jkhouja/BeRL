#!/usr/bin/env python3
"""Converter for CaSiNo (camp-site negotiation) — kchawla123/casino."""

from scripts.dialogue_converter_base import DialogueConverterBase


class CasinoConverter(DialogueConverterBase):
    DATA_SOURCE = "casino"

    def download_dataset(self):
        from datasets import load_dataset

        ds = load_dataset("kchawla123/casino", split=self.split)
        conversations = []
        for idx, row in enumerate(ds):
            dialog, speakers = [], []
            for log in row["chat_logs"]:
                text = (log.get("text") or "").strip()
                if not text or log.get("id") == "Submit-Deal":
                    continue
                dialog.append(text)
                speakers.append(log.get("id", f"agent_{len(dialog) % 2 + 1}"))
            if len(dialog) >= 2:
                conversations.append({
                    "dialog": dialog,
                    "speakers": speakers,
                    "conv_id": f"casino_{idx}",
                    "meta": {},
                })
        print(f"Loaded {len(conversations)} CaSiNo conversations")
        return conversations
