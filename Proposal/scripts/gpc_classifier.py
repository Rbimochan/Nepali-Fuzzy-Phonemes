"""Gaussian Process Classification per confusable pair (uncertainty-aware).

Core of my contribution: trains a GPC (RBF kernel, Laplace approximation
via scikit-learn — a dependency-light stand-in for GPflow at pilot scale,
same approximation the plan called for) per pair, and reports leave-one-out
accuracy plus posterior probability/variance per held-out clip. The
variance column is what later gets checked against human-ambiguity ratings
(the calibration check) and what feeds the shared disambiguation schema.

Usage:
    python Proposal/scripts/gpc_classifier.py \
        --manifest Proposal/artifacts/feature_manifest.csv \
        --out Proposal/artifacts/gpc_predictions.csv
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

FEATURE_PREFIXES = ("mfcc", "duration_s", "spectral_centroid_hz", "vot_proxy_s")


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith(FEATURE_PREFIXES)]


def posterior_variance(proba_row: np.ndarray) -> float:
    """Bernoulli-style variance of the top-class posterior; 0 = certain,
    0.25 = maximally uncertain (50/50) for a 2-class problem."""
    p = float(np.max(proba_row))
    return p * (1 - p)


def run_pair(df_pair: pd.DataFrame, feat_cols: list[str]) -> tuple[dict, list[dict]]:
    labels = sorted(df_pair["label"].unique())
    if len(labels) != 2 or len(df_pair) < 4:
        return {"n": len(df_pair), "accuracy": None, "note": "need exactly 2 labels, n>=4"}, []

    X = StandardScaler().fit_transform(df_pair[feat_cols].to_numpy())
    y = df_pair["label"].to_numpy()
    files = df_pair["file"].to_numpy()
    words = df_pair["word"].to_numpy()

    kernel = 1.0 * RBF(length_scale=1.0)
    loo = LeaveOneOut()

    preds, truths, rows = [], [], []
    for train_idx, test_idx in loo.split(X):
        clf = GaussianProcessClassifier(kernel=kernel, random_state=0)
        clf.fit(X[train_idx], y[train_idx])

        proba = clf.predict_proba(X[test_idx])[0]
        pred_label = clf.classes_[np.argmax(proba)]
        variance = posterior_variance(proba)

        i = test_idx[0]
        rows.append({
            "pair_id": df_pair["pair_id"].iloc[0],
            "file": files[i],
            "word": words[i],
            "true_label": y[i],
            "predicted_label": pred_label,
            "posterior_probability": float(np.max(proba)),
            "posterior_variance": variance,
        })
        preds.append(pred_label)
        truths.append(y[i])

    accuracy = accuracy_score(truths, preds)
    return {"n": len(y), "accuracy": accuracy, "note": "leave-one-out GPC"}, rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="Proposal/artifacts/feature_manifest.csv")
    ap.add_argument("--out", default="Proposal/artifacts/gpc_predictions.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    feat_cols = feature_columns(df)

    print(f"Loaded {len(df)} clips across {df['pair_id'].nunique()} pair(s).\n")

    all_rows = []
    for pair_id, df_pair in df.groupby("pair_id"):
        summary, rows = run_pair(df_pair, feat_cols)
        acc_str = f"{summary['accuracy']:.2f}" if summary["accuracy"] is not None else "n/a"
        print(f"[{pair_id}] n={summary['n']} accuracy={acc_str} ({summary['note']})")
        all_rows.extend(rows)

    if all_rows:
        out_df = pd.DataFrame(all_rows)
        out_df.to_csv(args.out, index=False)
        print(f"\nWrote {len(out_df)} per-clip predictions to {args.out}")
        print("\nPer-clip posterior (schema: pair_id, predicted_label, "
              "posterior_probability, posterior_variance):")
        print(out_df[["pair_id", "word", "true_label", "predicted_label",
                       "posterior_probability", "posterior_variance"]]
              .to_string(index=False))


if __name__ == "__main__":
    main()
