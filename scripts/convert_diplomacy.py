#!/usr/bin/env python3
"""Converter for Diplomacy (deception) — ConvoKit diplomacy-corpus."""

from scripts.convert_convokit_base import ConvoKitConverterBase


class DiplomacyConverter(ConvoKitConverterBase):
    CORPUS_ID = "diplomacy-corpus"
    DATA_SOURCE = "diplomacy"
