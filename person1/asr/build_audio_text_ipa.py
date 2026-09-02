#!/usr/bin/env python3
"""
build_audio_text_ipa.py -- Build TSV with audio path, reference text, and IPA phonemes.

Reads audio_index.tsv, generates IPA for each reference text using Piper/espeak-ng,
and writes a TSV suitable for downstream processing.

Usage:
  python build_audio_text_ipa.py
  python build_audio_text_ipa.py --limit 100    # test on first 100
"""

import argparse
import csv
import os
import re
import sys
import unicodedata

STRESS = re.compile(r"[\u02C8\u02CC]")
PUNCT = re.compile(r"[\.,;:!?\u0964\u0965]+")


def normalize_affricates(ipa):
    mapping = [
        ("tʃʰ", "t͡ʃʰ"), ("tsʰ", "t͡sʰ"), ("dʒʰ", "d͡ʒʰ"),
        ("tʃ", "t͡ʃ"), ("ts", "t͡s"), ("dʒ", "d͡ʒ"), ("dz", "d͡z"),
    ]
    for plain, breve in mapping:
        ipa = ipa.replace(plain, breve)
    return ipa


def phonemize_text(ph, text, voice="ne"):
    words = text.split()
    ipa_words = []
    for w in words:
        w_clean = unicodedata.normalize("NFC", w.strip())
        if not w_clean:
            continue
        try:
            result = ph.phonemize(voice, w_clean)
            ipa = "".join("".join(ph_list) for ph_list in result)
            ipa = PUNCT.sub("", ipa)
            ipa = re.sub(r"\s+", " ", ipa).strip()
            ipa = STRESS.sub("", ipa)
            ipa = normalize_affricates(ipa)
            ipa_words.append(ipa)
        except Exception:
            ipa_words.append("")
    return " ".join(ipa_words)


def main():
    ap = argparse.ArgumentParser(description="Build audio/text/IPA TSV.")
    ap.add_argument("--index", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "audio_index.tsv"))
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "audio_text_ipa.tsv"))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    from piper.phonemize_espeak import EspeakPhonemizer
    ph = EspeakPhonemizer()

    rows = []
    with open(args.index, encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            rows.append(row)

    if args.limit:
        rows = rows[:args.limit]

    print(f"Generating IPA for {len(rows)} utterances...", file=sys.stderr)

    cache = {}
    out_rows = []
    for i, row in enumerate(rows):
        text = row["reference"]
        if text not in cache:
            cache[text] = phonemize_text(ph, text)
        ipa = cache[text]
        out_rows.append([row["audio_location"], text, ipa])

        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{len(rows)} ({len(cache)} unique texts)",
                  file=sys.stderr)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["audio_path", "text", "ipa"])
        w.writerows(out_rows)

    print(f"Done: {len(out_rows)} rows, {len(cache)} unique texts. "
          f"Wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
