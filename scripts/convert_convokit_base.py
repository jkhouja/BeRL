#!/usr/bin/env python3
"""Shared base for ConvoKit-backed converters.

Uses utterance-id insertion order (not timestamps) to avoid crashes on
corpora with unset timestamps (e.g. persuasionforgood).
"""

from typing import List, Dict, Any

from scripts.dialogue_converter_base import DialogueConverterBase


class ConvoKitConverterBase(DialogueConverterBase):
    CORPUS_ID = None
    DATA_SOURCE = "convokit"

    def _conv_split(self, conv) -> str:
        return conv.meta.get("split", "train")

    def _keep_utt(self, utt) -> bool:
        return not utt.meta.get("is_section_header", False)

    def download_dataset(self) -> List[Dict[str, Any]]:
        from convokit import Corpus, download

        print(f"Downloading {self.CORPUS_ID} via convokit...")
        corpus = Corpus(filename=download(self.CORPUS_ID))
        conversations = []
        for conv in corpus.iter_conversations():
            if self._conv_split(conv) != self.split:
                continue
            dialog, speakers = [], []
            for uid in conv.get_utterance_ids():
                utt = corpus.get_utterance(uid)
                text = (utt.text or "").strip()
                if not text or not self._keep_utt(utt):
                    continue
                dialog.append(text)
                speakers.append(str(utt.speaker.id))
            if len(dialog) >= 2:
                conversations.append({
                    "dialog": dialog,
                    "speakers": speakers,
                    "conv_id": conv.id,
                    "meta": {},
                })
        print(f"Loaded {len(conversations)} {self.DATA_SOURCE} conversations (split='{self.split}')")
        return conversations
