"""Fuzzy c-means membership scores per confusable pair.

Reads the feature manifest, standardizes features, runs fuzzy c-means
(c=2, one cluster per label in the pair) via scikit-fuzzy, and reports
per-clip membership scores plus a hard-label accuracy derived from
argmax membership (for comparison against the RF baseline).

Usage:
    python Proposal/scripts/fuzzy_classifier.py \
        --manifest Proposal/artifacts/feature_manifest.csv
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import skfuzzy as fuzz
from sklearn.preprocessing import StandardScaler

FEATURE_PREFIXES = ("mfcc", "duration_s", "spectral_centroid_hz", "vot_proxy_s")


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith(FEATURE_PREFIXES)]


def run_pair(df_pair: pd.DataFrame, feat_cols: list[str]) -> dict:
    labels = sorted(df_pair["label"].unique())
    if len(labels) != 2 or len(df_pair) < 4:
        return {"n": len(df_pair), "accuracy": None, "note": "need exactly 2 labels, n>=4"}

    X = StandardScaler().fit_transform(df_pair[feat_cols].to_numpy())

    # skfuzzy expects features x samples
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        X.T, c=2, m=2.0, error=1e-5, maxiter=1000, seed=0
    )

    # u has shape (c, n_samples); assign each cluster to the label whose
    # true members have the highest average membership in that cluster.
    y_true = df_pair["label"].to_numpy()
    cluster_to_label = {}
    for c_idx in range(2):
        avg_by_label = {
            lab: u[c_idx, y_true == lab].mean() if (y_true == lab).any() else -1
            for lab in labels
        }
        cluster_to_label[c_idx] = max(avg_by_label, key=avg_by_label.get)

    hard_pred = [cluster_to_label[c] for c in u.argmax(axis=0)]
    accuracy = float(np.mean(np.array(hard_pred) == y_true))

    return {
        "n": len(df_pair),
        "accuracy": accuracy,
        "fpc": float(fpc),
        "note": f"fuzzy c-means (fpc={fpc:.2f})",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="Proposal/artifacts/feature_manifest.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    feat_cols = feature_columns(df)

    print(f"Loaded {len(df)} clips across {df['pair_id'].nunique()} pair(s).\n")

    for pair_id, df_pair in df.groupby("pair_id"):
        result = run_pair(df_pair, feat_cols)
        acc_str = f"{result['accuracy']:.2f}" if result["accuracy"] is not None else "n/a"
        print(f"[{pair_id}] n={result['n']} accuracy={acc_str} ({result['note']})")


if __name__ == "__main__":
    main()
