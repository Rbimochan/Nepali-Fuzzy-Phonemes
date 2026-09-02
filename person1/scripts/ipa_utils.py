#!/usr/bin/env python3
"""IPA normalization helpers shared by the Person-1 pipeline.

espeak-ng writes affricates as plain digraphs (ts, dʒ, tʃ, ...) but PanPhon's
segment inventory keys them with a combining breve (t͡s, d͡ʒ, t͡ʃ, ...). Without
normalization, panphon.ipa_segs would split an affricate into a stop + fricative
pair, inflating word length and distorting the phonetic edit distance.

Longest forms first so that aspirated affricates are mapped before the plain
digraph match.
"""

AFFRICATE_MAP = [
    ("tʃʰ", "t͡ʃʰ"),
    ("tsʰ", "t͡sʰ"),
    ("dʒʰ", "d͡ʒʰ"),
    ("tʃ", "t͡ʃ"),
    ("ts", "t͡s"),
    ("dʒ", "d͡ʒ"),
    ("dz", "d͡z"),
]


def normalize_affricates(ipa):
    """Map espeak-ng plain affricate digraphs to PanPhon breve forms."""
    for plain, breve in AFFRICATE_MAP:
        ipa = ipa.replace(plain, breve)
    return ipa