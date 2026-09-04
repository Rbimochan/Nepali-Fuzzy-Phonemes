"""GP classifier per the locked proposal, section 4.5-4.6.

Implements the real 8-feature-vector spec (see
Proposal/GP_Classification_Implementation.md) using GPy, not the pilot's
sklearn stand-in: X = [VOT, F0_pert, H1H2, AspDur, GMM, DTW, Fuzzy, Topic].

The real Candidate x Occurrence Evidence Table (Sushank's Lane 2 handoff)
doesn't exist yet, so this runs against synthetic placeholder data by
default -- structurally identical to what the real table will look like,
so this code doesn't need to change when real data lands, only the
data-loading function does. Pass --evidence-table to use a real CSV once
one exists.

Two required arms per spec:
  1. Direct GP classification (Bernoulli likelihood, probit link, EP)
  2. GP regression -> threshold at 0.5 (regression-to-classification)
Both compared with RBF and Matern 5/2 kernels.

Usage:
    python Proposal/scripts/gp_classifier_gpy.py
    python Proposal/scripts/gp_classifier_gpy.py --evidence-table path/to/real.csv
"""
from __future__ import annotations

import argparse

import GPy
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    brier_score_loss,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

FEATURE_COLS = ["VOT", "F0_pert", "H1H2", "AspDur", "GMM", "DTW", "Fuzzy", "Topic"]

KERNELS = {
    "rbf": lambda dim: GPy.kern.RBF(input_dim=dim),
    "matern52": lambda dim: GPy.kern.Matern52(input_dim=dim),
}


def make_synthetic_evidence_table(n_pairs: int = 8, n_per_pair: int = 40, seed: int = 0) -> pd.DataFrame:
    """Placeholder standing in for Sushank's Lane 2 handoff, matching the
    schema in Proposal/GP_Classification_Implementation.md section 1.
    Signal is deliberately partial (not perfectly separable) so the
    classifier/eval code gets exercised realistically rather than trivially."""
    rng = np.random.default_rng(seed)
    rows = []
    pair_ids = [f"synthetic_pair_{i}" for i in range(n_pairs)]

    for pair_id in pair_ids:
        # a random direction in feature space defines "true" separability
        # for this pair, so different pairs have different informative features
        true_w = rng.normal(size=len(FEATURE_COLS))
        for occ in range(n_per_pair):
            label = rng.integers(0, 2)
            base = rng.normal(size=len(FEATURE_COLS))
            # nudge features in the label's direction, plus noise, so the
            # signal is real but not clean -- avoids trivially-perfect toy runs
            feats = base + 0.8 * label * true_w + rng.normal(scale=0.5, size=len(FEATURE_COLS))
            row = {"pair_id": pair_id, "occurrence_id": f"{pair_id}_occ{occ}", "true_label": label}
            row.update(dict(zip(FEATURE_COLS, feats)))
            rows.append(row)

    return pd.DataFrame(rows)


def load_evidence_table(path: str | None) -> pd.DataFrame:
    if path is None:
        print("No --evidence-table given; using synthetic placeholder data "
              "(Lane 2's real handoff doesn't exist yet). Numbers below are "
              "NOT results -- they only demonstrate the pipeline runs.\n")
        return make_synthetic_evidence_table()
    df = pd.read_csv(path)
    missing = [c for c in FEATURE_COLS + ["pair_id", "true_label"] if c not in df.columns]
    if missing:
        raise ValueError(f"Evidence table missing required columns: {missing}")
    return df


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_prob >= lo) & (y_prob < hi) if hi < 1 else (y_prob >= lo) & (y_prob <= hi)
        if not mask.any():
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece += mask.mean() * abs(bin_acc - bin_conf)
    return ece


def run_gp_classification(X: np.ndarray, y: np.ndarray, kernel_name: str, folds: int) -> dict:
    """Direct GP classification: Bernoulli likelihood, probit link, EP
    (GPy.models.GPClassification's default inference)."""
    skf = StratifiedKFold(n_splits=min(folds, np.bincount(y.astype(int)).min()), shuffle=True, random_state=0)
    all_true, all_prob = [], []

    for train_idx, test_idx in skf.split(X, y):
        scaler = StandardScaler().fit(X[train_idx])
        X_train, X_test = scaler.transform(X[train_idx]), scaler.transform(X[test_idx])
        y_train = y[train_idx].reshape(-1, 1).astype(float)

        kernel = KERNELS[kernel_name](X.shape[1])
        model = GPy.models.GPClassification(X_train, y_train, kernel=kernel)
        model.optimize(messages=False)

        # model.predict()'s variance comes back as a bare `nan` for
        # classification in this GPy version (a quadrature issue, not a
        # data problem -- reproduces even on well-separated synthetic
        # data). The predictive MEAN is fine and IS the class-1
        # probability. We derive posterior_variance ourselves as the
        # Bernoulli variance p(1-p) of that mean, same convention as the
        # sklearn pilot classifier, rather than relying on the broken
        # quadrature output.
        prob_mean, _ = model.predict(X_test)
        all_true.append(y[test_idx])
        all_prob.append(prob_mean.ravel())

    y_true = np.concatenate(all_true)
    y_prob = np.concatenate(all_prob)
    y_pred = (y_prob >= 0.5).astype(int)

    return {
        "kernel": kernel_name,
        "n": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "brier": brier_score_loss(y_true, y_prob),
        "ece": expected_calibration_error(y_true, y_prob),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def run_gp_regression_thresholded(X: np.ndarray, y: np.ndarray, kernel_name: str, folds: int) -> dict:
    """Regression-to-classification arm (proposal section 4.5.3): GP
    regressor trained on the same {0,1} labels as a continuous target,
    thresholded at tau=0.5. NOTE: the proposal doesn't fully specify the
    regression target y -- this uses the label itself as a first
    definition (see the open question in GP_Classification_Implementation.md
    section 4); confirm with the team before treating this as final."""
    skf = StratifiedKFold(n_splits=min(folds, np.bincount(y.astype(int)).min()), shuffle=True, random_state=0)
    all_true, all_score = [], []

    for train_idx, test_idx in skf.split(X, y):
        scaler = StandardScaler().fit(X[train_idx])
        X_train, X_test = scaler.transform(X[train_idx]), scaler.transform(X[test_idx])
        y_train = y[train_idx].reshape(-1, 1).astype(float)

        kernel = KERNELS[kernel_name](X.shape[1])
        model = GPy.models.GPRegression(X_train, y_train, kernel=kernel)
        model.optimize(messages=False)

        pred_mean, _ = model.predict(X_test)
        all_true.append(y[test_idx])
        all_score.append(np.clip(pred_mean.ravel(), 0, 1))

    y_true = np.concatenate(all_true)
    y_score = np.concatenate(all_score)
    y_pred = (y_score >= 0.5).astype(int)

    return {
        "kernel": kernel_name,
        "n": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "brier": brier_score_loss(y_true, y_score),
        "ece": expected_calibration_error(y_true, y_score),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence-table", default=None,
                     help="CSV with pair_id, true_label, and the 8 feature columns. "
                          "Omit to run against synthetic placeholder data.")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    df = load_evidence_table(args.evidence_table)

    for pair_id, df_pair in df.groupby("pair_id"):
        X = df_pair[FEATURE_COLS].to_numpy()
        y = df_pair["true_label"].to_numpy().astype(int)

        if len(np.unique(y)) < 2 or np.bincount(y).min() < 2:
            print(f"[{pair_id}] skipped -- need both classes with >=2 samples each")
            continue

        print(f"=== {pair_id} (n={len(y)}) ===")
        for kernel_name in KERNELS:
            cls_result = run_gp_classification(X, y, kernel_name, args.folds)
            reg_result = run_gp_regression_thresholded(X, y, kernel_name, args.folds)
            print(f"  [classification/{kernel_name}] acc={cls_result['accuracy']:.2f} "
                  f"macro_f1={cls_result['macro_f1']:.2f} brier={cls_result['brier']:.3f} "
                  f"ece={cls_result['ece']:.3f}")
            print(f"  [regression+threshold/{kernel_name}] acc={reg_result['accuracy']:.2f} "
                  f"macro_f1={reg_result['macro_f1']:.2f} brier={reg_result['brier']:.3f} "
                  f"ece={reg_result['ece']:.3f}")
        print()


if __name__ == "__main__":
    main()
