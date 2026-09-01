#!/usr/bin/env python3
"""
phoneme_db.py -- Person 1: build the standard word+phoneme database.

Inputs (artifacts, must exist):
  vocab.tsv      word, freq, n_utts, n_speakers        (from word_stats.py)
  phonemes.tsv   word, ipa, status                     (from phonemize.py)
  utt_spk_text.tsv                                    (raw SLR54 index)

For every word it stores:
  - the word itself + source ("slr54"; extensible to other corpora)
  - its IPA (G2P via phonemize.py) and phonological segments
  - per-segment PanPhon articulatory feature vectors (24-dim each)
  - frequency, speaker count
  - every OpenSLR utterance ID where the word occurs (occurrence index)

It also builds the lookup buckets used by similar_words.py:
  key = (source, n_segments, first_segment) -> [word, ...]

Outputs (in db/ -- the database is the primary Person-1 deliverable):
  db/phoneme_db.tsv            word, source, ipa, segments, n_segments,
                               freq, n_utts, n_speakers
  db/phoneme_occurrences.tsv   word, source, n_utts, file_ids (comma-sep)
  db/phoneme_db.pkl            full dict + buckets + metadata
"""

import argparse
import collections
import csv
import os
import pickle
import sys
import unicodedata

import panphon

from ipa_utils import normalize_affricates
from phonetic_distance import segment_feats
from word_stats import is_devanagari_word, normalize_token

FEATURE_SET = "spe+"
PAD_LEN = 20  # fixed vector length used for the ANN-free exact lookup


def load_vocab(path):
    """word -> (freq, n_utts, n_speakers)"""
    out = {}
    with open(path, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            out[r["word"]] = (int(r["freq"]), int(r["n_utts"]), int(r["n_speakers"]))
    return out


def load_phonemes(path):
    """word -> ipa (status==ok only)"""
    out = {}
    with open(path, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            if r.get("status") == "ok":
                out[r["word"]] = unicodedata.normalize("NFD", r["ipa"].strip())
    return out


def load_occurrences(index_path):
    """word -> set(file_id), using the same token normalization as word_stats."""
    occ = collections.defaultdict(set)
    with open(index_path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            file_id, transcript = parts[0], "\t".join(parts[2:])
            for tok in transcript.split():
                tok = normalize_token(tok)
                if is_devanagari_word(tok):
                    occ[tok].add(file_id)
    return occ


def main():
    ap = argparse.ArgumentParser(description="Build the word+phoneme database.")
    ap.add_argument("--vocab", default="artifacts/vocab.tsv")
    ap.add_argument("--phonemes", default="artifacts/phonemes.tsv")
    ap.add_argument("--index", default="data/slr54/utt_spk_text.tsv")
    ap.add_argument("--db-dir", default="db",
                    help="output dir for the phoneme DB (default: db)")
    ap.add_argument("--source", default="slr54", help="corpus source name")
    args = ap.parse_args()

    os.makedirs(args.db_dir, exist_ok=True)
    ft = panphon.FeatureTable(feature_set=FEATURE_SET)

    vocab = load_vocab(args.vocab)
    phonemes = load_phonemes(args.phonemes)
    occurrences = load_occurrences(args.index)
    print(f"vocab={len(vocab)} phonemes={len(phonemes)} "
          f"occurrences-words={len(occurrences)}", file=sys.stderr)

    entries = {}          # (source, word) -> dict
    word_index = collections.defaultdict(list)  # word -> [(source, entry)]
    buckets = collections.defaultdict(list)     # (source, n_seg, first_seg) -> [word]

    db_rows = []
    occ_rows = []

    for word, (freq, n_utts, n_speakers) in vocab.items():
        ipa = phonemes.get(word)
        if not ipa:
            continue
        segs = ft.ipa_segs(normalize_affricates(ipa))
        if not segs:
            continue
        feats = segment_feats(ft, segs)  # numeric 24-dim vectors per segment
        file_ids = sorted(occurrences.get(word, ()))

        entry = {
            "word": word,
            "source": args.source,
            "ipa": normalize_affricates(ipa),  # pre-normalized for fast lookup
            "ipa_orig": ipa,
            "segs": segs,
            "feats": feats,
            "freq": freq,
            "n_utts": len(file_ids),
            "n_speakers": n_speakers,
            "file_ids": file_ids,
        }
        key = (args.source, word)
        entries[key] = entry
        word_index[word].append((args.source, entry))
        buckets[(args.source, len(segs), segs[0])].append(word)

        db_rows.append([word, args.source, ipa, " ".join(segs), len(segs),
                        freq, len(file_ids), n_speakers])
        occ_rows.append([word, args.source, len(file_ids), ",".join(file_ids)])

    db_path = os.path.join(args.db_dir, "phoneme_db.tsv")
    with open(db_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["word", "source", "ipa", "segments", "n_segments",
                    "freq", "n_utts", "n_speakers"])
        w.writerows(db_rows)

    occ_path = os.path.join(args.db_dir, "phoneme_occurrences.tsv")
    with open(occ_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["word", "source", "n_utts", "file_ids"])
        w.writerows(occ_rows)

    payload = {
        "feature_set": FEATURE_SET,
        "feature_names": ft.names,
        "pad_len": PAD_LEN,
        "source": args.source,
        "entries": {f"{s}::{w}": e for (s, w), e in entries.items()},
        "word_index": dict(word_index),
        "buckets": {f"{s}::{n}::{f}": ws for (s, n, f), ws in buckets.items()},
    }
    pkl_path = os.path.join(args.db_dir, "phoneme_db.pkl")
    with open(pkl_path, "wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"DB: {len(entries)} words, {len(buckets)} buckets, "
          f"{sum(len(w) for w in buckets.values())} bucket entries", file=sys.stderr)
    print(f"Wrote:\n  {db_path}\n  {occ_path}\n  {pkl_path}")


if __name__ == "__main__":
    main()
