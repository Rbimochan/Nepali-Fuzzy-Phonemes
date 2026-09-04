# GP Classification Implementation — Person C (Bimochan)

Implementation spec for §4.5–4.6 of the locked proposal
(`Fuzzy_Phonemes_Nepali_ASR_Disambiguation.pdf`, March 2026). This is the
build plan for Lane 3 ("Probabilistic Disambiguation") in Figure 1 of that
paper — the Gaussian Process classifier that consumes Sushank's
candidate-level hybrid evidence table and produces final predictions with
calibrated uncertainty.

**Status:** planning/spec document. The pilot work done so far
(`Proposal/scripts/*.py`, tracked in `personc.html`) is a *different,
earlier* feature set (MFCC/centroid/duration/VOT-proxy) built before this
proposal was locked — it validated that a record → extract → classify
pipeline runs end to end, but does not implement the 8-feature vector
below. This doc is the plan for building the real thing.

---

## 1. Input contract — what Lane 3 receives from Lane 2

Per Figure 4 of the proposal, Sushank's evidence-fusion stage hands off a
**Candidate × Occurrence Evidence Table**. Lane 3 code should treat this
as an external interface, not something it computes itself:

| Column | Source | Notes |
|---|---|---|
| `occurrence_id` | Lane 1 (Utsab) | links back to the ASR hypothesis/speech segment |
| `candidate_word` | Lane 1 | one row per (occurrence, candidate) pair |
| `true_label` | ground truth / pilot annotation | which candidate was actually spoken |
| `VOT` | Lane 1 phonetic-cue extraction | ms |
| `F0_pert` | Lane 1 | Hz |
| `H1H2` | Lane 1 | dB, spectral tilt |
| `AspDur` | Lane 1 | ms |
| `GMM` | Lane 2 | GMM log-likelihood score |
| `DTW` | Lane 2 | DTW similarity score |
| `Fuzzy` | Lane 2 | aggregated fuzzy membership score (§4.3.3) |
| `Topic` | Lane 2 | LDA context-compatibility score (§4.4) |

**This table doesn't exist yet** — it depends on Utsab's phonetic-cue
extraction (VOT/F0pert/H1H2/AspDur via Praat) and Sushank's GMM/DTW/fuzzy/
LDA evidence branches, none of which are built. Lane 3 work is legitimately
blocked on this handoff for real data, but the classifier and evaluation
code can be written and tested against synthetic placeholder rows now —
this is the "toy/dummy outputs" parallelization Utsab suggested in the Aug
20 discussion, so Lane 3 isn't idle waiting.

## 2. Feature vector (§4.5.1)

```
X = [VOT, F0_pert, H1H2, AspDur, GMM, DTW, Fuzzy, Topic]
```

8-dimensional, satisfying the coursework's "≥4 input variables" requirement
with margin. All 8 columns standardized to zero mean/unit variance using
**training-split statistics only** (§5.1.2 — never fit the scaler on
val/test, that's leakage).

## 3. GP classifier (§4.5.2)

| Spec | Value |
|---|---|
| Task | Binary, candidate A vs candidate B per confusable pair |
| Likelihood | Bernoulli, probit link, expectation propagation |
| Kernel (compare both) | Squared Exponential (RBF): `k(x,x') = σ_f² exp(-‖x-x'‖²/2ℓ²)` |
| | Matérn 5/2: `k(x,x') = σ_f²(1 + √5‖x-x'‖/ℓ + 5‖x-x'‖²/3ℓ²) exp(-√5‖x-x'‖/ℓ)` |
| Hyperparameter fitting | Marginal likelihood maximization, L-BFGS-B |
| Implementation | `GPy` (not the pilot's `sklearn.gaussian_process` — GPy is the library the proposal names, has native EP/probit support and multi-kernel comparison built in) |

**Action:** add `GPy` to `Proposal/scripts/requirements.txt` alongside the
existing `scikit-learn` (sklearn's GPC stays useful for quick sanity checks,
but the deliverable classifier should be GPy per spec).

## 4. Regression-to-classification (§4.5.3)

Second required arm, satisfying the coursework's "derive classification
from regression" requirement:

1. Train a GP **regressor** (same kernel choices) on a continuous
   confusability score `y ∈ [0, 1]`.
2. Threshold at `τ = 0.5` for binary predictions.
3. Compare against the direct GP classifier on the same metrics.

`y` needs a definition before this can be built — proposal doesn't fully
specify it. Reasonable default: `y = 1` for the true candidate, `0`
otherwise, i.e. train the regressor on the same labels as the classifier
and let the GP produce a continuous score instead of a class probability.
Flag this as an open question to confirm with the team, not something to
silently assume.

## 5. Baselines (§4.6) — now Sushank's per the workload rebalance

The proposal specifies SVM (RBF, grid search), Random Forest (100 trees,
Gini), and Logistic Regression (L2, cross-validated C), all on the same
8-D vector. Per the Aug-20-later workload conversation, **running these is
Sushank's task**, not Bimochan's — Bimochan's lane stays "GP framework,
kernel comparison, uncertainty analysis" per the Author Contributions
section. This doc notes the spec for interface compatibility (Lane 3's
evaluation code needs to consume Sushank's baseline output in the same
format as its own GP predictions) but the implementation itself is out of
scope here.

## 6. Evaluation (owned by Bimochan)

Per the proposal's framing and the earlier group discussion (results
should read as *planned evaluation*, not claimed findings, until real
experiments have run):

- **Candidate-level**: accuracy, macro-F1, confusion matrix
- **Calibration**: Brier score, Expected Calibration Error (ECE) — this is
  the paper's actual novelty claim per the abstract, keep this rigorous
- **ASR-level**: targeted substitution error rate, WER before/after
  correction, CER
- **Ablation grid (E0–E7)**: 8 configurations toggling feature subsets
  on/off (general-only, cue-only, context-only, full hybrid, etc.) —
  *design* is Bimochan's, *execution/rerunning* is Sushank's per the
  rebalance (§3 of Figure 6)

**Do not report specific numbers (e.g. "23.7% relative error reduction",
"Brier = 0.072") anywhere in real documentation until an actual experiment
has produced them.** Those numbers in the locked proposal PDF are
illustrative/target figures from before any code existed — repeating them
as if they're results would misrepresent unrun work.

## 7. Build plan

1. **Done** — `Proposal/scripts/gp_classifier_gpy.py`: GPy `GPClassification`
   (Bernoulli/probit/EP) and `GPRegression`+threshold, both with RBF and
   Matérn 5/2 kernels, stratified k-fold CV, accuracy/macro-F1/Brier/ECE/
   confusion matrix, run against `make_synthetic_evidence_table()` (same
   schema as the real handoff) or a real CSV via `--evidence-table`.
   One real bug hit and worked around: GPy's `.predict()` returns the
   predictive variance as a bare `nan` for classification in this version
   (a quadrature issue, reproduces even on well-separated data) — the
   predictive *mean* is fine and is the class probability, so
   `posterior_variance` is derived as `p(1-p)` instead of trusting the
   broken quadrature output.
2. **Done** — evaluation harness (accuracy/F1/Brier/ECE/confusion matrix)
   is part of the same script, run per-pair.
3. **TODO:** wire this into the ablation loop structure from Figure 6
   (E0–E7 feature-subset toggles) — not yet built, straightforward given
   `FEATURE_COLS` is already a flat list to subset.
4. **Blocked on Utsab:** VOT/F0pert/H1H2/AspDur extraction per real clip.
4. **Blocked on Sushank:** GMM/DTW/Fuzzy/Topic columns in the evidence
   table.
5. **Once 3+4 land:** swap synthetic data for real, rerun, report real
   numbers — this is when §6 of the paper gets filled in for real.
6. Resolve the open question in §4 above (definition of the regression
   target `y`) with the team before or during step 1.

## 8. Relationship to the existing pilot

The pilot pipeline (`extract_features.py`, `baseline_classifier.py`,
`fuzzy_classifier.py`, `gpc_classifier.py`, all in `Proposal/scripts/`)
stays useful as a *standalone* proof that a self-recorded-clip → feature →
classifier loop works, and its 81-clip dataset is real, usable data — but
it's a different feature representation than this spec and shouldn't be
confused with the Lane 3 deliverable described above. Once the real
8-feature evidence table exists, this pilot's clips could still be
folded in as additional training data if the audio and word list overlap
with the team's OpenSLR-based confusable-pair set.
