#!/usr/bin/env python3
"""
similar_words.py -- Person 1: fast "similar-sounding words" lookup.

Mechanism (length-bucketed exact, no ANN, no padding):
  For a query word of length L (in segments) and onset S, the candidates are
  every word in the same-onset buckets for lengths L-1, L, L+1. Each candidate
  is scored with the exact PanPhon feature-edit distance (normalized by max
  length) -- the same metric that validated well on the corpus.

  The query may be ANY Devanagari string: known words use their precomputed
  features, and unknown / misspelled words (ASR hypotheses, typos, nonce
  forms) are tokenized on the fly with espeak-ng G2P + PanPhon (WordTokenizer)
  and matched against the same buckets. This is the core design goal: map any
  word to its closest CORRECT pronunciations so the appropriate word can be
  selected.

Use:
  python similar_words.py --word कल --k 5
  python similar_words.py --word जिम्मेबारि --k 5    # misspelled/OOV: still works
  python similar_words.py --dump --max-words 2000      # export similar_words.tsv

Importable:
  from similar_words import SimilarityIndex
  idx = SimilarityIndex("db/phoneme_db.pkl")
  idx.get_similar("कल", k=5)          -> [(word, dist, ipa, freq), ...]
  idx.get_similar("जिम्मेबारि", k=5)  -> correct words closest to the typo
  idx.get_similar_many(["कल", "सात"], k=5, jobs=4)
                                     -> {word: [(...), ...], ...}
"""

import argparse
import csv
import os
import pickle
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    from tqdm import tqdm
except ImportError:  # graceful fallback if tqdm is not installed
    tqdm = None

import panphon.distance

from ipa_utils import normalize_affricates
from phonemize import PUNCT, STRESS
from phonetic_distance import (batch_feature_edit_distance_div_maxlen,
                               segment_feats)
from word_stats import is_devanagari_word, normalize_token


class WordTokenizer:
    """On-the-fly G2P + segmenter for out-of-vocabulary / misspelled queries.

    Replicates the build-time pipeline (phonemize.py -> phoneme_db.py) so an
    arbitrary Devanagari string -- including ASR misrecognitions, typos, or
    nonce forms -- maps to the same IPA segments and numeric feature vectors
    used to build the DB, and can be compared against it.

    espeak-ng (via piper) is rule-based, so it phonemizes any string. panphon
    and piper are imported lazily, only when a word needs on-the-fly
    tokenization (the known-word fast path never pays this cost).
    """

    def __init__(self, voice="ne"):
        self.voice = voice
        self._phonemizer = None
        self._feature_table = None

    def _espeak(self):
        if self._phonemizer is None:
            from piper.phonemize_espeak import EspeakPhonemizer
            self._phonemizer = EspeakPhonemizer()
        return self._phonemizer

    def _ft(self):
        if self._feature_table is None:
            import panphon
            self._feature_table = panphon.FeatureTable()
        return self._feature_table

    def tokenize(self, word):
        """Return (segs, feats) for an arbitrary word, or None if it cannot
        be tokenized (non-Devanagari, empty G2P, no segments)."""
        if not is_devanagari_word(word):
            return None
        word = normalize_token(word)
        result = self._espeak().phonemize(self.voice, word)
        ipa = "".join("".join(ph) for ph in result)
        ipa = PUNCT.sub("", ipa)
        ipa = re.sub(r"\s+", " ", ipa).strip()
        ipa = STRESS.sub("", ipa)
        if not ipa:
            return None
        ipa = normalize_affricates(ipa)
        ft = self._ft()
        segs = ft.ipa_segs(ipa)
        if not segs:
            return None
        return segs, segment_feats(ft, segs)


class SimilarityIndex:
    """Loads phoneme_db.pkl once and answers similar-word queries."""

    def __init__(self, db_path, source=None):
        self.db_path = db_path
        with open(db_path, "rb") as fh:
            self.db = pickle.load(fh)
        self.source = source or self.db["source"]
        self.dist = panphon.distance.Distance()  # kept for legacy/single pairs
        self.tokenizer = WordTokenizer()
        # decode serialized dicts (keys "src::word", "src::len::firstseg")
        self.entries = {}
        for k, e in self.db["entries"].items():
            _, word = k.split("::", 1)
            self.entries[word] = e
        self.buckets = {}
        for k, ws in self.db["buckets"].items():
            src, n, first = k.split("::")
            self.buckets[(src, int(n), first)] = ws

    def lookup(self, word):
        """Return the entry dict for a word, or None."""
        return self.entries.get(word)

    def _candidates(self, n_segs, onset, len_radius=1):
        """Words in the same-onset buckets for lengths L±len_radius."""
        out = []
        for n in range(n_segs - len_radius, n_segs + len_radius + 1):
            if n < 1:
                continue
            out.extend(self.buckets.get((self.source, n, onset), ()))
        return out

    def candidate_words(self, entry, len_radius=1):
        return self._candidates(len(entry["segs"]), entry["segs"][0], len_radius)

    def get_similar(self, word, k=5, len_radius=1, threshold=None):
        """Top-k most confusable words for `word`.

        Known words use their precomputed features (fast path). Unknown or
        misspelled words -- e.g. ASR hypotheses -- are tokenized on the fly
        (espeak-ng G2P + PanPhon) and matched against the same buckets, so any
        Devanagari string maps to its closest correct words in the corpus.

        If the word is in the DB, it appears as the first result (dist=0).
        Returns list of dicts: word, dist, ipa, freq, n_utts, n_speakers,
        source, file_ids.
        """
        entry = self.lookup(word)
        if entry is not None:
            segs = entry["segs"]
            query_feats = entry["feats"]
        else:
            tok = self.tokenizer.tokenize(word)
            if not tok:
                return []
            segs, query_feats = tok
        candidates = [c for c in self._candidates(len(segs), segs[0], len_radius)
                      if c != word]
        # If word is in DB, prepend it as exact match (dist=0)
        results = []
        if entry is not None:
            results.append({
                "word": word,
                "dist": 0.0,
                "ipa": entry["ipa_orig"],
                "freq": entry["freq"],
                "n_utts": entry["n_utts"],
                "n_speakers": entry["n_speakers"],
                "source": entry["source"],
                "file_ids": entry["file_ids"],
            })
        if not candidates:
            return results[:k]
        cand_entries = [self.entries[c] for c in candidates]
        dists = batch_feature_edit_distance_div_maxlen(
            query_feats, [e["feats"] for e in cand_entries])

        scores = []
        for c, e, d in zip(candidates, cand_entries, dists):
            if threshold is not None and d > threshold:
                continue
            scores.append({
                "word": c,
                "dist": round(float(d), 4),
                "ipa": e["ipa_orig"],
                "freq": e["freq"],
                "n_utts": e["n_utts"],
                "n_speakers": e["n_speakers"],
                "source": e["source"],
                "file_ids": e["file_ids"],
            })
        scores.sort(key=lambda s: s["dist"])
        return (results + scores)[:k]

    def get_similar_many(self, words, k=5, len_radius=1, threshold=None,
                         jobs=None):
        """Batch similar-word lookup.

        Args:
            words: iterable of query words (known OR misspelled/OOV -- both
                   are matched; OOV words are tokenized on the fly).
            k, len_radius, threshold: same semantics as get_similar().
            jobs: if > 1, run in parallel worker processes (picklable
                  interface; the DB is re-loaded once per worker).

        Returns:
            {word: [similar dicts, ...]} preserving input order. Words that
            cannot be tokenized (or have no similar words) are omitted.
        """
        words = list(words)
        if jobs and jobs > 1 and len(words) > 1:
            return _get_similar_many_parallel(self, words, k, len_radius,
                                              threshold, jobs)
        out = {}
        for w in words:
            r = self.get_similar(w, k=k, len_radius=len_radius,
                                 threshold=threshold)
            if r:
                out[w] = r
        return out


def _batch_worker(db_path, words, k, radius, threshold):
    """Module-level worker for parallel batch search (picklable)."""
    idx = SimilarityIndex(db_path)
    out = {}
    for w in words:
        r = idx.get_similar(w, k=k, len_radius=radius, threshold=threshold)
        if r:
            out[w] = r
    return out


def _get_similar_many_parallel(idx, words, k, radius, threshold, jobs):
    """Parallel batch search; restores input order."""
    jobs = min(jobs, len(words))
    chunks = [words[i::jobs] for i in range(jobs)]
    merged = {}
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(_batch_worker, idx.db_path, c, k, radius,
                               threshold) for c in chunks]
        for fut in as_completed(futures):
            merged.update(fut.result())
    return {w: merged[w] for w in words if w in merged}


def _dump_worker(db_path, words, k, radius, threshold):
    """Process one chunk of query words; returns (n_words, output rows)."""
    idx = SimilarityIndex(db_path)
    rows = []
    for word in words:
        for rank, s in enumerate(idx.get_similar(word, k=k, len_radius=radius,
                                                 threshold=threshold), 1):
            rows.append([word, rank, s["word"], s["dist"], s["ipa"],
                         s["freq"], s["n_utts"], s["n_speakers"],
                         s["source"], len(s["file_ids"])])
    return len(words), rows


def load_word_list(path):
    with open(path, encoding="utf-8") as fh:
        return [w for w in fh.read().split() if w]


def main():
    ap = argparse.ArgumentParser(description="Similar-sounding word lookup.")
    ap.add_argument("--db", default="db/phoneme_db.pkl")
    ap.add_argument("--word", default=None, help="query word")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--radius", type=int, default=1,
                    help="segment-length window L±radius for candidates")
    ap.add_argument("--threshold", type=float, default=None,
                    help="max normalized distance to include")
    ap.add_argument("--dump", action="store_true",
                    help="export similar_words.tsv for the whole DB")
    ap.add_argument("--max-words", type=int, default=None,
                    help="limit words when --dump (for quick runs)")
    ap.add_argument("--jobs", type=int, default=None,
                    help="parallel workers for --dump (default: cpu count)")
    ap.add_argument("--out", default="artifacts/similar_words.tsv")
    args = ap.parse_args()

    idx = SimilarityIndex(args.db)
    print(f"Loaded {len(idx.entries)} words, {len(idx.buckets)} buckets",
          file=sys.stderr)

    if args.dump:
        words = sorted(idx.entries)
        if args.max_words:
            words = words[: args.max_words]
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        jobs = args.jobs or os.cpu_count() or 1
        jobs = min(jobs, len(words))
        # Many small interleaved chunks: even load + a smooth progress bar.
        n_chunks = max(1, min(jobs * 10, len(words)))
        chunks = [words[i::n_chunks] for i in range(n_chunks)]
        chunks = [c for c in chunks if c]

        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = [pool.submit(_dump_worker, args.db, chunk, args.k,
                                   args.radius, args.threshold) for chunk in chunks]
            with open(args.out, "w", encoding="utf-8", newline="") as fh:
                w = csv.writer(fh, delimiter="\t")
                w.writerow(["word", "rank", "similar_word", "dist", "ipa", "freq", "n_utts", "n_speakers", "source", "n_file_ids"])
                n = 0
                bar = tqdm(total=len(words), unit="words", desc="similar words",
                           dynamic_ncols=True) if tqdm is not None else None
                for fut in as_completed(futures):
                    cnt, rows = fut.result()
                    w.writerows(rows)
                    n += len(rows)
                    if bar is not None:
                        bar.update(cnt)
                if bar is not None:
                    bar.close()
        print(f"Wrote {args.out} ({len(words)} query words, {n} rows)",
              file=sys.stderr)
        return

    if not args.word:
        ap.error("--word is required unless --dump is used")

    entry = idx.lookup(args.word)
    if entry is not None:
        ipa_str = entry["ipa"]
    else:
        tok = idx.tokenizer.tokenize(args.word)
        ipa_str = " ".join(tok[0]) if tok else "?"
    results = idx.get_similar(args.word, k=args.k, len_radius=args.radius,
                              threshold=args.threshold)
    print(f"Similar to '{args.word}' ({ipa_str}):")
    for s in results:
        print(f"  {s['word']:<12} d={s['dist']:.4f}  {s['ipa']:<24} "
              f"freq={s['freq']} utts={s['n_utts']}")


if __name__ == "__main__":
    main()