#!/usr/bin/env python3
"""Converter for PersuasionForGood — ConvoKit persuasionforgood-corpus."""

from scripts.convert_convokit_base import ConvoKitConverterBase


class PersuasionForGoodConverter(ConvoKitConverterBase):
    CORPUS_ID = "persuasionforgood-corpus"
    DATA_SOURCE = "persuasionforgood"
