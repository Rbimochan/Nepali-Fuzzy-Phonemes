#!/usr/bin/env python3
"""
phonetic_distance.py -- vectorized PanPhon feature edit distance.

Implements the SAME recurrence as panphon.distance.Distance.feature_edit_distance
(min_edit_distance with unweighted del/ins/sub costs) but vectorized across many
candidate words at once with numpy, so a whole bucket of candidates is scored in
one batched dynamic program instead of one slow Python loop per pair.

Costs (must match panphon exactly):
  del_cost(v) = sum(0.5 if f == 0 else 1 for f in v) / len(v)
  ins_cost(v) = same as del_cost
  sub_cost(a, b) = sum(|a-b| / 2 for a,b in zip(a,b)) / len(a)

Distance = min_edit_distance over segment feature vectors, normalized by the
length of the longer word.
"""

import numpy as np

import panphon

ft = panphon.FeatureTable()


def _del_ins_cost(feats):
    """Per-segment insertion/deletion cost, vectorized. feats: (..., 24)."""
    return (np.count_nonzero(feats, axis=-1)
            + 0.5 * (feats == 0).sum(axis=-1)) / feats.shape[-1]


def to_numeric(segment_vector):
    """Convert panphon's string feature vector ('-','+','0') to floats."""
    return np.array([1.0 if x == "+" else (-1.0 if x == "-" else 0.0)
                     for x in segment_vector], dtype=np.float64)


def segment_feats(ft_local, segs):
    """segments -> list of numeric 24-dim feature vectors."""
    return [to_numeric(ft_local.segment_to_vector(s)) for s in segs]


def batch_feature_edit_distance(query_feats, candidates):
    """Batched feature-edit distance (raw) from a query to many candidates.

    query_feats: list of 24-dim feature vectors (the query word).
    candidates:  list of lists of 24-dim feature vectors (one per candidate).
    Returns:     numpy array of raw feature-edit distances, same order as
                 candidates. NaNs are replaced with inf (zero-length guard).
    """
    q = np.asarray(query_feats, dtype=np.float64)          # (n, 24)
    n = q.shape[0]
    q_del = _del_ins_cost(q)                                # (n,)

    # Group candidates by length so each DP batch has fixed target length.
    by_len = {}
    for ci, cand in enumerate(candidates):
        if len(cand) == 0:
            continue
        by_len.setdefault(len(cand), []).append(ci)

    raw = np.full(len(candidates), np.inf)

    for m, indices in by_len.items():
        feats = np.array([candidates[i] for i in indices], dtype=np.float64)  # (B, m, 24)
        B = len(indices)
        ins = _del_ins_cost(feats)                           # (B, m)

        # d[0][j] = cumulative insertion cost along target row.
        prev = np.concatenate([np.zeros((B, 1)), np.cumsum(ins, axis=1)], axis=1)  # (B, m+1)

        for i in range(1, n + 1):
            qi = q[i - 1]
            del_i = q_del[i - 1]
            sub = (np.abs(feats - qi).sum(axis=-1) / 2.0) / 24.0   # (B, m)
            cur = np.empty((B, m + 1))
            cur[:, 0] = prev[:, 0] + del_i                        # d[i][0]
            # Sequential over target columns (row-wise dependency), vectorized over B.
            for j in range(1, m + 1):
                top = prev[:, j] + del_i                          # d[i-1][j] + del
                diag = prev[:, j - 1] + sub[:, j - 1]             # d[i-1][j-1] + sub
                left = cur[:, j - 1] + ins[:, j - 1]              # d[i][j-1] + ins
                cur[:, j] = np.minimum(np.minimum(top, diag), left)
            prev = cur

        raw[indices] = prev[:, -1]

    return raw


def batch_feature_edit_distance_div_maxlen(query_feats, candidates):
    """Batched feature-edit distance normalized by max(query, candidate) length.

    Matches panphon's feature_edit_distance_div_maxlen for every candidate.
    """
    raw = batch_feature_edit_distance(query_feats, candidates)
    qn = len(query_feats)
    denom = np.array([max(qn, len(c)) for c in candidates], dtype=np.float64)
    return np.divide(raw, denom, out=np.full_like(raw, np.inf), where=denom > 0)


def validate():
    """Check the batched implementation against panphon's per-pair function."""
    import panphon.distance
    import random

    dist = panphon.distance.Distance()
    words = ["kaːl", "kʌl", "kʰʌl", "kaːn", "sʌt", "saːtʰ", "jʌtɪ", "jʌdɪ",
             "kʊl", "kɛːl", "kuːl", "t͡saːl", "saːs", "sʌtsəʈʈʰiː"]
    ft_local = ft
    seglist = [ft_local.ipa_segs(w) for w in words]
    feats = [segment_feats(ft_local, segs) for segs in seglist]

    errors = 0
    for i, (w1, f1) in enumerate(zip(words, feats)):
        for j, (w2, f2) in enumerate(zip(words, feats)):
            expect = dist.feature_edit_distance_div_maxlen(w1, w2)
            got = batch_feature_edit_distance_div_maxlen(f1, [f2])[0]
            if abs(expect - got) > 1e-9:
                errors += 1
                print(f"MISMATCH {w1} vs {w2}: panphon={expect} batched={got}")
    print(f"validate: {len(words) * len(words)} pairs, {errors} mismatches")
    return errors == 0


if __name__ == "__main__":
    import sys
    sys.exit(0 if validate() else 1)