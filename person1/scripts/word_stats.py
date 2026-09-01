#!/usr/bin/env python3
"""
word_stats.py -- Person 1, step P1.1: vocabulary statistics from OpenSLR SLR54.

Reads utt_spk_text.tsv (FileID \\t UserID \\t transcript) and computes:
  - token vocabulary with corpus frequencies
  - per-word speaker coverage (number of distinct speakers using the word)
  - per-word utterance coverage
  - corpus-level summary (utterances, speakers, tokens, vocabulary size)

Only Devanagari-bearing tokens are kept as vocabulary entries; the counts for
non-Devanagari tokens (Latin, digits, punctuation-only) are reported separately.

Usage:
  python word_stats.py --index ../data/slr54/utt_spk_text.tsv \
                       --out-dir ../outputs/p1
"""

import argparse
import collections
import csv
import os
import re
import unicodedata

DEVANAGARI = re.compile(r"[\u0900-\u097F]")
DANDA = re.compile(r"^[\u0964\u0965]+|[\u0964\u0965]+$")  # danda / double danda
ZWJ_ZWNJ = str.maketrans({"\u200D": "", "\u200C": ""})  # zero-width joiner / non-joiner


def normalize_token(tok):
    """NFC-normalize, drop zero-width joiners and stray punctuation."""
    tok = unicodedata.normalize("NFC", tok.strip())
    tok = tok.translate(ZWJ_ZWNJ)
    tok = DANDA.sub("", tok)
    return tok


def is_devanagari_word(tok):
    return bool(DEVANAGARI.search(tok))


def main():
    ap = argparse.ArgumentParser(description="Vocabulary statistics from SLR54 index.")
    ap.add_argument("--index", default="../data/slr54/utt_spk_text.tsv")
    ap.add_argument("--out-dir", default="../outputs/p1")
    ap.add_argument("--min-freq", type=int, default=1,
                    help="minimum corpus frequency to keep a word in the vocab table")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    word_freq = collections.Counter()
    word_speakers = collections.defaultdict(set)
    word_utts = collections.defaultdict(set)
    non_devanagari = collections.Counter()

    n_utts = 0
    speakers = set()

    with open(args.index, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            file_id, user_id, transcript = parts[0], parts[1], "\t".join(parts[2:])
            n_utts += 1
            speakers.add(user_id)
            for tok in transcript.split():
                tok = normalize_token(tok)
                if not tok:
                    continue
                if is_devanagari_word(tok):
                    word_freq[tok] += 1
                    word_speakers[tok].add(user_id)
                    word_utts[tok].add(file_id)
                else:
                    non_devanagari[tok] += 1

    rows = []
    for word, freq in word_freq.items():
        if freq < args.min_freq:
            continue
        rows.append({
            "word": word,
            "freq": freq,
            "n_utts": len(word_utts[word]),
            "n_speakers": len(word_speakers[word]),
        })
    rows.sort(key=lambda r: (-r["freq"], r["word"]))

    vocab_path = os.path.join(args.out_dir, "vocab.tsv")
    with open(vocab_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["word", "freq", "n_utts", "n_speakers"],
                           delimiter="\t")
        w.writeheader()
        w.writerows(rows)

    summary = {
        "utterances": n_utts,
        "speakers": len(speakers),
        "tokens": sum(word_freq.values()),
        "vocab_raw": len(word_freq),
        "vocab_kept": len(rows),
        "non_devanagari_types": len(non_devanagari),
        "non_devanagari_tokens": sum(non_devanagari.values()),
    }
    summary_path = os.path.join(args.out_dir, "corpus_stats.tsv")
    with open(summary_path, "w", encoding="utf-8") as fh:
        for k, v in summary.items():
            fh.write(f"{k}\t{v}\n")

    print("Corpus summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"Vocab table: {vocab_path}")
    print(f"Top 10 words: {', '.join(r['word'] for r in rows[:10])}")


if __name__ == "__main__":
    main()