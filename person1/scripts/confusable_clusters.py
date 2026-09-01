#!/usr/bin/env python3
"""
confusable_clusters.py -- Person 1, steps P1.5-P1.7: confusable word clusters
via PanPhon phonological feature edit distance.

Pipeline:
  1. Load a phonemized vocabulary TSV (word, ipa, status) optionally joined
     with frequencies (word_stats output).
  2. Tokenize each IPA string into phonological segments.
  3. Canopy filter: only words with the same length (within --max-len-diff)
     and the same first segment are compared, so cost stays near O(V).
  4. Pairwise PanPhon feature edit distance (normalized by max length).
  5. Edges below --threshold connect words; connected components are the
     confusable clusters.
  6. Write clusters + a pairs table + per-cluster stats.

The PanPhon reference is Mortensen et al. 2016 (COLING), "PanPhon: A Resource
for Mapping IPA Segments to Articulatory Feature Vectors" [C16-1328].

Usage:
  python confusable_clusters.py --phonemes ../outputs/p1/phonemes.tsv \
      --vocab ../outputs/p1/vocab.tsv \
      --out-dir ../outputs/p1 \
      --threshold 0.15
"""

import argparse
import csv
import os
import sys
import unicodedata

try:
    from tqdm import tqdm
except ImportError:  # graceful fallback if tqdm is not installed
    tqdm = None

import panphon
import panphon.distance

from ipa_utils import normalize_affricates


def norm_ipa(s):
    return unicodedata.normalize("NFD", s.strip())


def load_phonemes(path, vocab_path=None, min_freq=1, max_vocab=None):
    """Load word -> (ipa, status) plus optional (freq, n_speakers)."""
    freq = {}
    if vocab_path:
        with open(vocab_path, encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                f = int(row["freq"])
                if f >= min_freq:
                    freq[row["word"]] = f

    items = []  # (word, ipa_segments, freq)
    with open(path, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            if row.get("status") != "ok":
                continue
            word = row["word"]
            if vocab_path and word not in freq:
                continue
            ipa = norm_ipa(row["ipa"])
            if not ipa:
                continue
            items.append((word, ipa, freq.get(word, 0)))
    if max_vocab:
        items = items[: max_vocab]
    return items


def main():
    ap = argparse.ArgumentParser(description="PanPhon confusable-cluster discovery.")
    ap.add_argument("--phonemes", required=True, help="TSV from phonemize.py")
    ap.add_argument("--vocab", default=None, help="TSV from word_stats.py (frequency gate)")
    ap.add_argument("--out-dir", default="../outputs/p1")
    ap.add_argument("--min-freq", type=int, default=1,
                    help="only consider words appearing >= this many times")
    ap.add_argument("--max-vocab", type=int, default=None,
                    help="only consider the first N phonemized words (quick runs)")
    ap.add_argument("--threshold", type=float, default=0.06,
                    help="max normalized feature-edit distance for a confusable edge")
    ap.add_argument("--max-len-diff", type=int, default=1,
                    help="max segment-length difference to compare (canopy)")
    ap.add_argument("--mode", choices=["all", "knn"], default="knn",
                    help="'all' keeps every pair below --threshold; 'knn' keeps "
                         "only the K nearest neighbors per word (breaks chains)")
    ap.add_argument("--k", type=int, default=3,
                    help="neighbors per word when --mode=knn")
    args = ap.parse_args()

    ft = panphon.FeatureTable()
    dist = panphon.distance.Distance()

    items = load_phonemes(args.phonemes, args.vocab,
                          min_freq=args.min_freq, max_vocab=args.max_vocab)
    print(f"Loaded {len(items)} words", file=sys.stderr)

    # Tokenize IPA into segments, build length + first-segment canopies.
    tokenized = []
    for word, ipa, f in items:
        segs = ft.ipa_segs(normalize_affricates(ipa))
        if not segs:
            continue
        tokenized.append((word, segs, f))
    print(f"Tokenized {len(tokenized)} words", file=sys.stderr)

    canopy = {}
    for word, segs, f in tokenized:
        key = (len(segs), segs[0])
        canopy.setdefault(key, []).append((word, segs, f))

    edges = set()  # frozenset pairs
    pair_rows = []
    n_compared = 0
    nearest = {}  # word -> [(dist, other)] for knn mode
    total_pairs = sum(len(b) * (len(b) - 1) // 2 for b in canopy.values())
    bar = tqdm(total=total_pairs, unit="pairs", desc="phonetic pairs",
               dynamic_ncols=True) if tqdm is not None else None
    for key, bucket in canopy.items():
        L = len(bucket)
        for i in range(L):
            wa, segs_a, fa = bucket[i]
            for j in range(i + 1, L):
                if bar is not None:
                    bar.update(1)
                wb, segs_b, fb = bucket[j]
                if abs(len(segs_a) - len(segs_b)) > args.max_len_diff:
                    continue
                n_compared += 1
                ia = "".join(segs_a)
                ib = "".join(segs_b)
                d = dist.feature_edit_distance_div_maxlen(ia, ib)
                if d <= args.threshold:
                    edges.add(frozenset((wa, wb)))
                    pair_rows.append({
                        "word_a": wa, "word_b": wb,
                        "dist": f"{d:.4f}",
                        "ipa_a": ia, "ipa_b": ib,
                        "freq_a": fa, "freq_b": fb,
                    })
                if args.mode == "knn":
                    nearest.setdefault(wa, []).append((d, wb))
                    nearest.setdefault(wb, []).append((d, wa))
    if bar is not None:
        bar.close()

    if args.mode == "knn":
        # Keep only the K closest neighbors per word (symmetric prune).
        keep = set()
        for w, nbrs in nearest.items():
            for d, o in sorted(nbrs)[: args.k]:
                if d <= args.threshold:
                    keep.add(frozenset((w, o)))
        pair_rows = [r for r in pair_rows if frozenset((r["word_a"], r["word_b"])) in keep]
        edges = keep

    print(f"Compared {n_compared} candidate pairs; {len(edges)} confusable edges",
          file=sys.stderr)

    # Connected components = clusters.
    adj = {}
    for wa, wb in edges:
        adj.setdefault(wa, set()).add(wb)
        adj.setdefault(wb, set()).add(wa)

    seen = set()
    clusters = []
    for word in adj:
        if word in seen:
            continue
        comp = []
        stack = [word]
        seen.add(word)
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nxt in adj[cur]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        clusters.append(sorted(comp))
    clusters.sort(key=lambda c: -len(c))

    os.makedirs(args.out_dir, exist_ok=True)

    pairs_path = os.path.join(args.out_dir, "confusable_pairs.tsv")
    with open(pairs_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(pair_rows[0].keys()) if pair_rows
                           else ["word_a", "word_b", "dist", "ipa_a", "ipa_b", "freq_a", "freq_b"],
                           delimiter="\t")
        w.writeheader()
        w.writerows(pair_rows)

    clusters_path = os.path.join(args.out_dir, "confusable_clusters.tsv")
    with open(clusters_path, "w", encoding="utf-8", newline="") as fh:
        fh.write("cluster_id\tn_words\twords\n")
        for cid, comp in enumerate(clusters, 1):
            fh.write(f"{cid}\t{len(comp)}\t{' '.join(comp)}\n")

    stats_path = os.path.join(args.out_dir, "cluster_stats.tsv")
    with open(stats_path, "w", encoding="utf-8", newline="") as fh:
        fh.write("n_clusters\tn_edges\tn_compared\tn_words_clustered\n")
        fh.write(f"{len(clusters)}\t{len(edges)}\t{n_compared}\t{len(adj)}\n")

    print(f"Found {len(clusters)} clusters")
    for i, c in enumerate(clusters[:10], 1):
        print(f"  {i:>3}. ({len(c)}) {' '.join(c[:12])}")
    print(f"Wrote: {pairs_path}\n      {clusters_path}\n      {stats_path}")


if __name__ == "__main__":
    main()