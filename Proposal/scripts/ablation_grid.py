"""E0-E7 ablation grid per the proposal's Figure 6 ("Baselines & Ablations"
stage) and abstract ("eight experimental configurations").

The abstract's threefold evidence split maps onto exactly 2^3=8
configurations by toggling three branches on/off:
  - general  = GMM, DTW               (general acoustic evidence)
  - cue      = VOT, F0_pert, H1H2, AspDur, Fuzzy
               (explicit phonetic cues + their fuzzy-fusion aggregate --
               Fuzzy is a fusion of the cue branch per section 4.3.3, not
               an independent evidence source in the abstract's threefold
               split, so it's grouped here rather than treated as a 4th
               branch)
  - context  = Topic                  (LDA contextual evidence)

E0 is a majority-class baseline (no GP, sanity floor) rather than an
empty feature set, since a 0-feature GP isn't meaningful. E1-E7 are the
7 non-empty branch subsets, E7 being the full hybrid model. This ordering
lets you read the general/cue/context split matching the group's earlier
discussion (E5=general+context, E6=cue+context, E7=full -- E7 vs E4
isolates context's marginal contribution, matching what they called
"context/fuzzy contributes the smallest marginal gain of the three
evidence branches").

Reuses run_gp_classification from gp_classifier_gpy.py so this doesn't
duplicate the GPy model code -- only which columns go into X changes
per configuration.

Usage:
    python Proposal/scripts/ablation_grid.py
    python Proposal/scripts/ablation_grid.py --evidence-table path/to/real.csv
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold

from gp_classifier_gpy import (
    expected_calibration_error,
    load_evidence_table,
    run_gp_classification,
)

BRANCHES = {
    "general": ["GMM", "DTW"],
    "cue": ["VOT", "F0_pert", "H1H2", "AspDur", "Fuzzy"],
    "context": ["Topic"],
}

CONFIGS = [
    ("E0", []),  # majority-class baseline, not a real feature set
    ("E1", ["general"]),
    ("E2", ["cue"]),
    ("E3", ["context"]),
    ("E4", ["general", "cue"]),
    ("E5", ["general", "context"]),
    ("E6", ["cue", "context"]),
    ("E7", ["general", "cue", "context"]),  # full hybrid
]


def columns_for(branch_names: list[str]) -> list[str]:
    cols: list[str] = []
    for b in branch_names:
        cols.extend(BRANCHES[b])
    return cols


def run_majority_baseline(y: np.ndarray, folds: int) -> dict:
    skf = StratifiedKFold(n_splits=min(folds, np.bincount(y).min()), shuffle=True, random_state=0)
    all_true, all_pred, all_prob = [], [], []
    for train_idx, test_idx in skf.split(np.zeros_like(y), y):
        clf = DummyClassifier(strategy="most_frequent").fit(np.zeros((len(train_idx), 1)), y[train_idx])
        pred = clf.predict(np.zeros((len(test_idx), 1)))
        prob = clf.predict_proba(np.zeros((len(test_idx), 1)))
        prob_pos = prob[:, 1] if prob.shape[1] > 1 else np.full(len(test_idx), float(pred[0]))
        all_true.append(y[test_idx])
        all_pred.append(pred)
        all_prob.append(prob_pos)
    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    y_prob = np.concatenate(all_prob)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "brier": brier_score_loss(y_true, y_prob),
        "ece": expected_calibration_error(y_true, y_prob),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence-table", default=None)
    ap.add_argument("--kernel", default="rbf", choices=["rbf", "matern52"],
                     help="Ablation runs one kernel by default to keep the "
                          "8-config x n-pairs grid fast; use gp_classifier_gpy.py "
                          "directly for the full kernel comparison.")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--out", default="Proposal/artifacts/ablation_results.csv")
    args = ap.parse_args()

    df = load_evidence_table(args.evidence_table)
    rows = []

    for pair_id, df_pair in df.groupby("pair_id"):
        y = df_pair["true_label"].to_numpy().astype(int)
        if len(np.unique(y)) < 2 or np.bincount(y).min() < 2:
            print(f"[{pair_id}] skipped -- need both classes with >=2 samples each")
            continue

        print(f"=== {pair_id} (n={len(y)}) ===")
        for config_name, branches in CONFIGS:
            if config_name == "E0":
                result = run_majority_baseline(y, args.folds)
                cols_used = "(none -- majority-class baseline)"
            else:
                cols = columns_for(branches)
                X = df_pair[cols].to_numpy()
                result = run_gp_classification(X, y, args.kernel, args.folds)
                cols_used = "+".join(branches)

            rows.append({
                "pair_id": pair_id, "config": config_name, "branches": cols_used,
                "accuracy": result["accuracy"], "macro_f1": result["macro_f1"],
                "brier": result["brier"], "ece": result["ece"],
            })
            print(f"  [{config_name}: {cols_used}] acc={result['accuracy']:.2f} "
                  f"macro_f1={result['macro_f1']:.2f} brier={result['brier']:.3f} "
                  f"ece={result['ece']:.3f}")
        print()

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out, index=False)
    print(f"Wrote {len(out_df)} rows to {args.out}")

    print("\n=== averaged across pairs ===")
    summary = out_df.groupby("config")[["accuracy", "macro_f1", "brier", "ece"]].mean()
    summary = summary.reindex([c for c, _ in CONFIGS])
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
