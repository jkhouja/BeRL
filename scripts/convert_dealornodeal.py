#!/usr/bin/env python3
"""Converter for DealOrNoDeal — mikelewis0/deal_or_no_dialog."""

from scripts.dialogue_converter_base import DialogueConverterBase


class DealOrNoDealConverter(DialogueConverterBase):
    DATA_SOURCE = "dealornodeal"

    def download_dataset(self):
        from datasets import load_dataset

        ds = load_dataset("mikelewis0/deal_or_no_dialog", split=self.split,
                          revision="refs/convert/parquet")
        conversations = []
        for idx, row in enumerate(ds):
            raw = row.get("dialogue")
            if not raw:
                continue
            dialog, speakers = [], []
            for chunk in raw.split("<eos>"):
                chunk = chunk.strip()
                if not chunk:
                    continue
                if chunk.startswith("YOU:") or chunk.startswith("THEM:"):
                    spk, text = chunk.split(":", 1)
                    text = text.strip()
                    if text and text != "<selection>":
                        dialog.append(text)
                        speakers.append("YOU" if spk == "YOU" else "THEM")
            if len(dialog) >= 2:
                conversations.append({
                    "dialog": dialog,
                    "speakers": speakers,
                    "conv_id": f"dealornodeal_{idx}",
                    "meta": {},
                })
        print(f"Loaded {len(conversations)} DealOrNoDeal conversations")
        return conversations
