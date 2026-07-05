#!/usr/bin/env python3
"""Converter for Switchboard (SwDA) — ConvoKit switchboard-corpus."""

from scripts.convert_convokit_base import ConvoKitConverterBase


class SwitchboardConverter(ConvoKitConverterBase):
    CORPUS_ID = "switchboard-corpus"
    DATA_SOURCE = "switchboard"
