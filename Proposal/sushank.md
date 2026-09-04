# Lane 2 Implementation Spec — Sushank (Hybrid Evidence Modeling)

Implementation spec for §4.3.1–4.4 of the locked proposal
(`Fuzzy_Phonemes_Nepali_ASR_Disambiguation.pdf`, March 2026), matching the
format of `Proposal/GP_Classification_Implementation.md` and
`Proposal/utsab.md`. Documents what Lane 2 needs to produce for the
Candidate × Occurrence Evidence Table that Lane 3 (Bimochan) consumes.
Written by the Lane 3 person as a documentation pass, not an
implementation — Sushank owns the actual build. **Nothing for this lane
exists yet in the repo** (confirmed: no GMM/DTW/fuzzy/LDA code anywhere),
so this is a from-scratch spec, not a gap analysis against existing work
the way `utsab.md` is.

---

## 1. Input — what this lane receives from Lane 1 (Utsab)

Per Figure 3, the "Validated Handoff Package": confusable word clusters,
OpenSLR occurrences, ASR results, general acoustic features, phonetic cue
features. **This package doesn't exist yet** — see `Proposal/utsab.md`
§3, items 1–3 are the blockers. Everything below can be built and tested
against synthetic placeholder data in the meantime, the same way
`gp_classifier_gpy.py` and `ablation_grid.py` already do on the Lane 3
side.

## 2. Four evidence branches to build (§4.3.1, §4.3.3, §4.4, Figure 4)

### 2.1 GMM likelihood scoring
Per §4.3.1: train a **Gaussian Mixture Model (3 components)** for each
candidate word using training data, using the general acoustic features
(MFCC/F0/energy/spectral/duration) Lane 1 provides. Score = log-likelihood
of the observed features under that candidate's GMM. Output: one `GMM`
score per (occurrence, candidate) row.

### 2.2 DTW similarity scoring
Dynamic Time Warping similarity between the observed acoustic segment and
reference templates per candidate. The proposal doesn't fully specify the
feature representation DTW runs on (raw MFCC frames is the standard
choice and most likely what's intended, given MFCCs are already being
computed for the GMM branch) — confirm this rather than guessing a
different representation. Output: one `DTW` score per row.

### 2.3 Fuzzy evidence integration (§4.3.3, Figure 5)
For each phonetic cue (VOT primarily, per Figure 5's example; extendable
to the other three cues once they exist), define fuzzy membership
functions from training-data distributions — the proposal's Figure 5
example uses three memberships (Unaspirated / Aspirated / Ambiguous) as
sigmoid/bell-shaped curves over VOT in ms. Aggregate per-cue membership
degrees into one `Fuzzy` score via a **weighted sum, weights learned
during training** (exact learning procedure for the weights isn't
specified in the proposal — needs a design decision, e.g. fit against
the same training labels the GP classifier will use, or a separate
calibration step).

### 2.4 LDA contextual evidence (§4.4)
1. **Corpus**: 50,000 Nepali documents (news, Wikipedia, web text) — not
   yet compiled anywhere in the repo.
2. **Preprocessing**: tokenization, stopword removal, Nepali stemming.
3. **Training**: LDA with **20 topics** (proposal states this was
   "determined by coherence analysis" — that analysis doesn't appear to
   exist yet either; running it, not just picking 20 by assumption, is
   part of this step), via `gensim`.
4. **Context scoring**: for each candidate word `w` and its surrounding
   sentence context (excluding the target word):
   ```
   ContextScore(w) = Σ(t=1 to K) P(t|context) · P(w|t)
   ```
   where `K=20`, `P(t|context)` is the posterior topic probability given
   the context, `P(w|t)` the topic-word probability. Output: one `Topic`
   score per row.

## 3. Output — the Candidate × Occurrence Evidence Table

Per Figure 4's "Candidate Evidence Handoff": one row per (candidate,
occurrence) with columns `GMM`, `DTW`, phonetic-cue scores, `Fuzzy`,
`Topic`. This is exactly the schema `Proposal/GP_Classification_Implementation.md`
§1 already specifies as Lane 3's input contract — **that's the file to
match column-for-column**, not a new schema to design independently:

| Column | Owner |
|---|---|
| `occurrence_id`, `candidate_word`, `true_label` | Lane 1 |
| `VOT`, `F0_pert`, `H1H2`, `AspDur` | Lane 1 |
| `GMM`, `DTW`, `Fuzzy`, `Topic` | **this lane** |

## 4. Additional tasks per the workload rebalance (Aug 20+ discussion)

Per the group's later conversation (moving load off Bimochan since the GP
lane "owns both the core model and all the comparison/evaluation
scaffolding"), these were reassigned to Sushank — noted here for
completeness since they're mechanical executions of Lane 3's code, not
new design:

1. **Baseline classifiers** (§4.6) — SVM (RBF, grid search), Random
   Forest (100 trees, Gini), Logistic Regression (L2, cross-validated).
   All on the same 8-D feature vector. Not yet implemented anywhere;
   `Proposal/scripts/gp_classifier_gpy.py` establishes the data-loading
   and CV pattern these should follow for consistency, but the baseline
   models themselves need to be written.
2. **GP Regression variant execution** (§4.5.3) — once Bimochan's
   `gp_classifier_gpy.py` pipeline is confirmed working (it is, on
   synthetic data — see `Proposal/GP_Classification_Implementation.md`
   §7), rerun it on real data once available; this is "reuse the GP
   pipeline with a different target," not new implementation.
3. **Ablation grid execution** (E0–E7) — `Proposal/scripts/ablation_grid.py`
   already exists and is verified working (both RBF and Matérn 5/2, on
   synthetic data). Running it on real data once it lands is this lane's
   task per the rebalance; the design/script itself is done.
4. **Bias/fairness subgroup reporting** (§7.2 of the paper) — splitting
   evaluation metrics by gender/speaker group. Not yet built; needs
   speaker metadata that should come through Lane 1's speaker-disjoint
   split definitions.

## 5. Open questions to resolve before/during implementation

- DTW's exact feature representation (§2.2 above).
- The fuzzy-weight learning procedure (§2.3 above).
- Whether the "20 topics, determined by coherence analysis" claim needs
  the coherence analysis actually run, or whether 20 is being taken as a
  reasonable default to state outright.
- Confirm this doc's understanding of the handoff-table schema
  (§3) against `Proposal/GP_Classification_Implementation.md` §1 with
  Bimochan before building, since a schema mismatch here would silently
  break Lane 3's evidence-table loading.
