#!/usr/bin/env python3
"""EDA stage 03 — Devanagari character-level stats from the SLR54 index.

Stats  : stats/chars/char_freq.tsv, char_class.tsv
Images : images/chars/char_freq_top.png
"""

import collections
import re
import unicodedata

from common import DATA_INDEX, savefig, setup_matplotlib, write_stats

plt = setup_matplotlib()

DEVANAGARI = re.compile(r"[\u0900-\u097F]")


def main():
    chars = collections.Counter()
    for line in open(DATA_INDEX, encoding="utf-8"):
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 3:
            continue
        transcript = "\t".join(parts[2:])
        chars.update(unicodedata.normalize("NFC", transcript))

    dev = {c: n for c, n in chars.items() if DEVANAGARI.match(c)}
    total = sum(dev.values())
    rows = sorted(dev.items(), key=lambda x: -x[1])
    table = [[c, n, round(100 * n / total, 3), unicodedata.name(c, "?")]
             for c, n in rows]
    write_stats("chars", "char_freq.tsv", ["char", "count", "pct", "unicode_name"], table)

    # Classify characters: vowels, consonants, matras (virama / vowel signs), digits, others
    def classify(c):
        code = ord(c)
        if "\u0966" <= c <= "\u096F":
            return "digit"
        if code in (0x094D,):
            return "virama"
        if 0x093E <= code <= 0x094C:
            return "vowel_sign"
        if 0x0904 <= code <= 0x0914:
            return "independent_vowel"
        if 0x0915 <= code <= 0x0939:
            return "consonant"
        if code in (0x0900, 0x0901, 0x0902, 0x0903):
            return "anunasika/anusvara"
        if code in (0x0964, 0x0965):
            return "danda"
        return "other"

    cls = collections.Counter()
    for c, n in dev.items():
        cls[classify(c)] += n
    write_stats("chars", "char_class.tsv", ["class", "count", "pct"],
                [[c, n, round(100 * n / total, 2)] for c, n in cls.most_common()])

    # Plot top 40 characters
    top = rows[:40]
    names = [c for c, _ in top]
    counts = [n for _, n in top]
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(top)), counts, color="#4C72B0")
    plt.xticks(range(len(top)), names)
    plt.xlabel("character")
    plt.ylabel("count")
    plt.title(f"Top 40 Devanagari characters ({total:,} total)")
    print("Wrote", savefig(plt, "chars", "char_freq_top.png"))


if __name__ == "__main__":
    main()