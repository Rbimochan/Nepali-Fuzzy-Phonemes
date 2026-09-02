#!/usr/bin/env python3
"""
phonemize.py -- Person 1, step P1.4: G2P (Devanagari -> IPA) using Piper's
bundled espeak-ng engine.

Reads a word list (whitespace/newline separated) and writes a TSV:
    word \\t ipa \\t status

The IPA output is espeak-ng's IPA transcription (voice language 'ne').
By default stress marks (U+02C8, U+02CC) are stripped so that segmental
similarity is compared; keep them with --keep-stress.

Usage:
  python phonemize.py --words ../outputs/p1/top_vocab.txt \
                      --out ../outputs/p1/phonemes.tsv
  python phonemize.py --words ../outputs/p1/vocab.tsv --column word ...
"""

import argparse
import csv
import os
import re
import sys
import unicodedata

STRESS = re.compile(r"[\u02C8\u02CC]")  # primary / secondary stress
PUNCT = re.compile(r"[\.,;:!?\u0964\u0965]+")


def load_words(path, column=None):
    """Return an ordered de-duplicated list of words."""
    words = []
    seen = set()
    if path.endswith(".tsv"):
        with open(path, encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                if column is None:
                    column = "word"
                w = row.get(column)
                if w and w not in seen:
                    seen.add(w)
                    words.append(w)
    else:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                for w in line.split():
                    w = unicodedata.normalize("NFC", w.strip())
                    if w and w not in seen:
                        seen.add(w)
                        words.append(w)
    return words


def main():
    ap = argparse.ArgumentParser(description="G2P Devanagari -> IPA via Piper/espeak-ng.")
    ap.add_argument("--words", required=True, help="input word list or TSV")
    ap.add_argument("--column", default="word",
                    help="column name when --words is a TSV")
    ap.add_argument("--out", required=True, help="output TSV (word, ipa, status)")
    ap.add_argument("--voice", default="ne", help="espeak-ng voice/language")
    ap.add_argument("--keep-stress", action="store_true",
                    help="keep primary/secondary stress marks in the IPA")
    ap.add_argument("--limit", type=int, default=None,
                    help="phonemize only the first N words (for quick runs)")
    args = ap.parse_args()

    from piper.phonemize_espeak import EspeakPhonemizer

    words = load_words(args.words, args.column)
    if args.limit:
        words = words[: args.limit]
    print(f"Phonemizing {len(words)} words with voice '{args.voice}'...", file=sys.stderr)

    ph = EspeakPhonemizer()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    n_ok = n_fail = 0
    with open(args.out, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["word", "ipa", "status"])
        for word in words:
            try:
                result = ph.phonemize(args.voice, word)
                ipa = "".join("".join(ph_list) for ph_list in result)
                ipa = PUNCT.sub("", ipa)
                ipa = re.sub(r"\s+", " ", ipa).strip()
                if not args.keep_stress:
                    ipa = STRESS.sub("", ipa)
                status = "ok" if ipa else "empty"
                n_ok += int(status == "ok")
                n_fail += int(status != "ok")
            except Exception as exc:
                ipa, status = "", f"error: {type(exc).__name__}"
                n_fail += 1
            w.writerow([word, ipa, status])

    print(f"Done: {n_ok} ok, {n_fail} failed/empty. Wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()