"""Hard-boundary baseline (Random Forest) per confusable pair.

Reads the feature manifest produced by extract_features.py, trains a
leave-one-out Random Forest per pair (small pilot sample sizes don't
support a held-out split yet), and reports per-pair accuracy.

Usage:
    python Proposal/scripts/baseline_classifier.py \
        --manifest Proposal/artifacts/feature_manifest.csv
"""
from __future__ import annotations

import argparse

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score

FEATURE_PREFIXES = ("mfcc", "duration_s", "spectral_centroid_hz", "vot_proxy_s")


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith(FEATURE_PREFIXES)]


def run_pair(df_pair: pd.DataFrame, feat_cols: list[str]) -> dict:
    X = df_pair[feat_cols].to_numpy()
    y = df_pair["label"].to_numpy()

    if len(set(y)) < 2 or len(y) < 4:
        return {"n": len(y), "accuracy": None, "note": "too few samples/classes"}

    loo = LeaveOneOut()
    preds, truths = [], []
    for train_idx, test_idx in loo.split(X):
        clf = RandomForestClassifier(n_estimators=200, random_state=0)
        clf.fit(X[train_idx], y[train_idx])
        preds.append(clf.predict(X[test_idx])[0])
        truths.append(y[test_idx][0])

    acc = accuracy_score(truths, preds)
    return {"n": len(y), "accuracy": acc, "note": "leave-one-out"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="Proposal/artifacts/feature_manifest.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    feat_cols = feature_columns(df)

    print(f"Loaded {len(df)} clips across {df['pair_id'].nunique()} pair(s), "
          f"{len(feat_cols)} features.\n")

    for pair_id, df_pair in df.groupby("pair_id"):
        result = run_pair(df_pair, feat_cols)
        acc_str = f"{result['accuracy']:.2f}" if result["accuracy"] is not None else "n/a"
        print(f"[{pair_id}] n={result['n']} accuracy={acc_str} ({result['note']})")


if __name__ == "__main__":
    main()
