# Nepali Fuzzy Phonemes — Complete Research Proposal

**Author:** Bimochan Kunwar
**Working title:** *Fuzzy Phonemes: Modeling Uncertainty in Nepali Phoneme Classification for Low-Resource Speech Recognition*
**Status:** Standalone blueprint — consolidates and extends the existing [`datacollection.md`](datacollection.md), [`Experimental_Paper_Blueprint_FuzzyPhonemes.md`](Experimental_Paper_Blueprint_FuzzyPhonemes.md), and the compiled bibliography ([`compass_artifact_..._text_markdown.md`](compass_artifact_wf-49b0073f-9161-5a8f-b990-faec71b76e2a_text_markdown.md)) into one document.

---

## 0. Central Thesis

The goal is **not** to build another Nepali phoneme classifier. The goal is to test a specific hypothesis:

> In real Nepali speech, some phonemes are not cleanly separable acoustic categories — they overlap, especially under dialectal variation, casual speech, and known confusable pairs (aspirated/unaspirated stops, dental/retroflex contrasts, vowel-length distinctions). Modeling this overlap explicitly, as **fuzzy or probabilistic membership** rather than a forced discrete label, should produce classifiers that are more accurate on ambiguous cases, better calibrated, and more useful as an uncertainty-aware signal inside a downstream Nepali ASR pipeline than a standard hard-decision classifier.

Everything below is organized to produce **evidence for or against that hypothesis**, not just a working system.

---

## 1. Research Problem and Research Questions

### 1.1 Problem statement

Nepali ASR systems (Whisper-Nepali fine-tunes, CNN-Transformer LVCSR, per [Paudel et al. 2023](https://aclanthology.org/2023.ldk-1.33/), [Ghimire et al. 2024](https://arxiv.org/abs/2402.03050)) still confuse phonetically close Devanagari words. Standard phoneme classifiers and G2P/ASR pipelines assume each phoneme occupies a discrete, mutually exclusive acoustic class. That assumption breaks down at known confusion boundaries in Nepali phonology (e.g. ब/भ aspirated-unaspirated, ट/त retroflex-dental, vowel-length pairs). No existing Nepali resource labels *how confusable* a phoneme pair is, nor treats that confusability as a first-class, learnable quantity.

### 1.2 Research questions

- **RQ1 (Existence):** Do measurable, systematic zones of acoustic/perceptual overlap exist between specific Nepali phoneme pairs, and can they be identified from real speech data rather than assumed from phonological theory alone?
- **RQ2 (Representation):** Can fuzzy-set membership functions or probabilistic (Bayesian/GP) posteriors represent this overlap in a way that is more faithful to the data than a hard classification boundary?
- **RQ3 (Performance):** Does an uncertainty-aware fuzzy/probabilistic phoneme classifier outperform a matched hard-boundary baseline on classification accuracy for confusable pairs specifically (not necessarily on average accuracy across all phonemes)?
- **RQ4 (Calibration):** Is the uncertainty produced by the fuzzy/probabilistic model *well-calibrated* — i.e., does high model uncertainty actually correlate with genuine acoustic ambiguity (validated against human/native-speaker disagreement), rather than being an arbitrary confidence score?
- **RQ5 (Downstream value):** Does exposing this uncertainty signal to a downstream disambiguation stage (word-level confusable-pair resolution, per the existing `datacollection.md` pipeline) reduce error rate on confusable-word cases relative to using only 1-best hard phoneme/ASR output?

### 1.3 Scope boundary

This proposal is phoneme- and word-pair-level. It deliberately does **not** attempt full end-to-end Nepali ASR retraining — it treats Whisper-Nepali / the LVCSR CNN-Transformer as a fixed upstream ASR front end, per the existing paper blueprint, and focuses the novel contribution on the fuzzy-phoneme representation and classification layer that sits on top.

---

## 2. Why This Matters

- **Low-resource ASR gap:** Nepali has ~30M+ speakers but a thin ASR research base ([Ghimire survey](https://arxiv.org/abs/2402.03050)); confusable-word errors disproportionately hurt low-resource systems because there isn't enough data to brute-force away ambiguity with scale, so an explicit uncertainty model is more valuable here than in high-resource languages.
- **Genuine literature gap:** The compiled bibliography shows the "fuzzy + phonetic + ASR" intersection (Pillar 1) is thin and dated (an SSRN 2019 paper, a 1996 IEEE decoder) — this is a real, defensible novelty gap, not a fabricated one.
- **Practical stakes:** Confusable-word errors change meaning, not just fluency — relevant to downstream Nepali voice applications (health-voice pipelines, accessibility tools, transcription for education/media).
- **Methodological contribution beyond Nepali:** A validated fuzzy/probabilistic phoneme-uncertainty framework, if it works, generalizes as a technique to other low-resource, phonologically dense languages (other Indic languages, tonal languages, etc.).

---

## 3. Definition and Linguistic Characterization of Nepali Fuzzy Phonemes

### 3.1 Working definition

A **fuzzy phoneme instance** is an acoustic realization whose feature representation lies close enough to the decision boundary between two (or more) phoneme categories that native-speaker listeners and/or acoustic classifiers disagree on its label above a defined disagreement threshold. Fuzziness is a *property of specific tokens/contexts*, not of the phoneme inventory as a whole — most Nepali phoneme instances will be unambiguous; the fuzzy set is the minority that sits near category boundaries.

### 3.2 Nepali phonological contrasts relevant to fuzziness

To be developed into a full characterization document during Layer 1, but scoped now:

| Contrast type | Example pairs (Devanagari) | Why it's fuzzy-prone |
|---|---|---|
| Aspirated vs. unaspirated stops | क/ख, ग/घ, ट/ठ, ड/ढ, प/फ, ब/भ | Voice Onset Time (VOT) is a continuous acoustic cue; casual/fast speech compresses VOT distinctions |
| Retroflex vs. dental stops | ट/त, ड/द, ण/न | Small articulatory distance; heavy dialectal variation in retroflexion degree |
| Vowel length | इ/ई, उ/ऊ | Duration-based contrast, easily neutralized in unstressed positions or fast speech |
| Nasalization (chandrabindu/anusvara) | plain vs. nasalized vowels | Coarticulatory nasal spread makes the boundary gradient, not binary |
| Sibilants | श/ष/स | Historically merged in many spoken dialects; near-total overlap in casual speech |
| Schwa deletion contexts | word-final/medial inherent vowel | Not a phoneme-identity fuzziness but a segmentation fuzziness — affects alignment, not just classification |

### 3.3 Deliverable for this section

A short **Nepali Fuzzy Phoneme Inventory** document (Layer 1 output): each candidate fuzzy pair, its phonological basis, expected acoustic cue, and a literature citation or native-speaker judgment supporting its inclusion. This becomes the seed list that Section 8 (dataset construction) operationalizes.

---

## 4. Identifying Confusable Phoneme Pairs/Groups

Two complementary identification strategies, both already partially scaffolded in `datacollection.md`:

1. **Top-down (phonological theory):** Use the contrast table above plus consultation of Nepali phonology references to hypothesize confusable pairs before touching data.
2. **Bottom-up (data-driven):** Extract vocabulary from the OpenSLR SLR54 corpus, run it through the in-house rule-based Devanagari G2P converter (Section 4 of `datacollection.md`), compute phonetic edit-distance over all word pairs, and surface close matches as *candidate* confusable pairs. Cross-reference against actual ASR substitution errors (1-best vs. reference transcript mismatches) mined from SLR54/Whisper-Nepali outputs — a substitution error where the ASR emitted a phonetically close word is strong empirical evidence of real confusability, stronger than edit-distance alone.

Native-speaker manual filtering (already planned) reconciles the two: keep pairs that are *both* theoretically plausible *and* empirically evidenced; discard edit-distance-close pairs that aren't actually confused in practice, and flag any bottom-up discoveries not predicted by theory as a specific finding worth reporting.

---

## 5. Literature Review Strategy

### 5.1 Structure (already scaffolded — 4 pillars in the bibliography)

- **Pillar 1 — Fuzzy + phonetic + ASR (core anchor, ~7 papers):** Bora et al. 2019 fuzzy phoneme recognition; 1996 fuzzy acoustic-phonetic decoder; Jain et al. 2025 hybrid deep-learning+fuzzy; Jain, Jindal & Jain 2024 fuzzy-graph Hindi-English correction; phonetic+semantic N-best correction (2022); Graph-Based Phonetic Error Correction (2026 preprint); PARCO (2025 preprint).
- **Pillar 2 — ASR post-processing / N-best re-ranking / homophone-confusion (mature field, any language, ~7 papers):** HyPoradise, N-best T5, SoftCorrect, Homophone Disambiguation in Speech Transformers, Soft-Masked BERT, Mandarin polyphone disambiguation (Cai et al. 2019), German acoustic-feature homophone disambiguation.
- **Pillar 3 — Nepali/Indic low-resource NLP & speech (~8 papers):** Ghimire survey, Whisper-Nepali fine-tuning, NepBERTa, Paudel et al. LVCSR, DPCSpell, Vartani Spellcheck, Vashishtha fuzzy-logic-in-text-mining review, Shahi & Shakya Nepali SMS classification.
- **Pillar 4 — Foundational ML/optimization/embeddings (~8 papers):** Zadeh (fuzzy sets), Levenshtein (edit distance — unverified citation, needs confirming before submission), Breiman (Random Forests), Mikolov (word2vec), Kim (text CNN), Sharma et al. (Phonetic Word Embeddings), Bayesian HPO, Radford et al. (Whisper).

### 5.2 New areas to add for the GPC/uncertainty-modeling extension of this proposal

The existing bibliography is strong on fuzzy logic and ASR post-processing but **thin on probabilistic/Bayesian uncertainty modeling and Gaussian Process Classification specifically** — this is the gap to fill before drafting the Layer 3 sections:

- Gaussian Process Classification foundations: Rasmussen & Williams (2006), *Gaussian Processes for Machine Learning* (canonical GP textbook — covers Laplace approximation, EP, and variational inference for GP classification).
- Scalable/sparse GP methods for larger datasets: Hensman et al. (2015) on variational sparse GPs (relevant once the fuzzy-pairs dataset grows beyond what exact GP inference handles cheaply).
- Uncertainty quantification and calibration in classification: Guo et al. (2017), *On Calibration of Modern Neural Networks* — needed for RQ4's calibration evaluation methodology.
- Fuzzy set theory vs. probability theory as competing uncertainty formalisms: a short comparative-methods citation (e.g., Zadeh's own later writing on possibility theory, or a survey contrasting fuzzy vs. Bayesian uncertainty) to justify testing both rather than picking one a priori.
- Phoneme classification with GPs or Bayesian methods specifically, if any prior work exists (search required — likely thin, which itself supports the novelty claim for Layer 3).
- Soft/fuzzy phoneme boundary detection in forced alignment tools (e.g., how tools like the Montreal Forced Aligner or wav2vec2-based aligners currently treat boundaries as hard cut points) — needed to justify Section 9's "fuzzy boundary" methodology.

**Action item:** run a dedicated literature search on "Gaussian Process Classification speech" / "Gaussian Process phoneme" / "uncertainty quantification low-resource ASR" before finalizing Layer 3, and add 3–5 verified citations to the bibliography under a new **Pillar 5 — Uncertainty & GP Methods**.

### 5.3 Search keywords to use

`fuzzy phoneme recognition`, `phoneme confusion matrix`, `phonetic confusability modeling`, `Gaussian Process Classification speech`, `Gaussian Process phoneme classification`, `uncertainty quantification ASR`, `Bayesian phoneme classification`, `soft phoneme boundaries`, `phoneme boundary detection forced alignment`, `low-resource speech recognition uncertainty`, `calibration speech classifiers`, `Nepali phonology acoustic`, `Devanagari grapheme-to-phoneme`, `polyphone disambiguation`, `homophone disambiguation ASR`, `speech perception categorical vs continuous`, `fuzzy set theory speech`.

---

## 6. Nepali Phonology, Phonetics, and Speech Characteristics

Required background to compile in Layer 1 (native-speaker-informed, cross-checked against published Nepali phonology references):

- Full Nepali phoneme inventory (consonants, vowels, diphthongs) with IPA mapping.
- Devanagari orthography-to-phoneme mapping rules, including inherent-vowel (schwa) deletion, conjunct consonants (संयुक्ताक्षर), and nasalization marks — this is exactly what `datacollection.md` Section 4 already scopes as the in-house G2P converter.
- Known dialectal variation across Nepal's regions (e.g., eastern vs. western Nepali, Terai-influenced varieties) and how it affects the fuzzy contrasts in Section 3.2.
- Prosodic characteristics (stress, intonation patterns) to the extent they interact with segmental confusability (e.g., unstressed-position neutralization).
- A short review of how Nepali compares to related Indic languages (Hindi, Bengali) on these same contrasts, since much of the closest fuzzy/phonetic literature (Sharma et al. Phonetic Word Embeddings, Jain et al. fuzzy-graph correction) is validated on Hindi — establishing where Nepali phonology matches or diverges from Hindi determines how directly those methods transfer.

---

## 7. First-Hand and Secondary Research Resources Required

### 7.1 Secondary (literature, existing datasets, existing tools)

- All papers in the 4 (soon 5) bibliography pillars — already collected as PDFs under `Papers/`.
- OpenSLR SLR54 (Nepali ASR transcribed speech), SLR43 (Nepali TTS), Mozilla Common Voice Nepali — already identified in `datacollection.md`.
- NepBERTa corpus (0.8B-word Nepali text) for vocabulary frequency and text-mining context features.
- Existing forced-alignment tools (Montreal Forced Aligner or a wav2vec2 CTC-based aligner) — needed for phoneme-level segmentation before any classification can happen.

### 7.2 First-hand (must be produced by this project)

- Native-speaker phonological judgments on candidate fuzzy pairs (Section 4).
- The purpose-built confusable-pairs audio dataset (Section 8, per `datacollection.md`).
- Human disagreement/ambiguity labels on a subset of fuzzy-boundary clips (needed for RQ4 calibration validation — see Section 15).
- The in-house rule-based Devanagari G2P converter (no existing Nepali pronunciation lexicon exists).

---

## 8. Dataset: Collection, Annotation, and Quality Control

This section **incorporates and extends** the existing `datacollection.md` in full; only the additions specific to *fuzzy* annotation are new here.

### 8.1 Two-layer corpus structure (existing plan, retained)

- **Layer 1 — Base corpus:** SLR54 (~157k utterances, ~9.3GB), SLR43 (~800MB TTS), Common Voice Nepali. Used for vocabulary/frequency statistics, background ASR, and substitution-error mining.
- **Layer 2 — Confusable-pairs dataset (the novel contribution):** purpose-built minimal-pair audio + metadata, target 300–500 clips initially, shaped as ~50 pairs × 2 words × 5–8 speakers rather than many pairs from one speaker (speaker variation stresses the confusability signal more than raw pair count).

### 8.2 New addition — fuzzy annotation layer

Beyond the existing per-pair schema (`pair_id`, `word_A`, `word_B`, `phoneme_A`, `phoneme_B`, `confusion_type`, `source`), add:

- **Per-clip ambiguity rating**, collected from 2–3 independent native-speaker annotators per clip: "which word did you hear?" (forced choice) plus a confidence score (e.g. 1–5 Likert). Disagreement across annotators, or low self-reported confidence, is the ground-truth signal for how "fuzzy" that specific token actually is — this is what calibration (RQ4) is checked against.
- **Inter-annotator agreement** computed per pair (e.g. Cohen's/Fleiss' kappa) — pairs with low agreement are, by definition, the genuinely fuzzy ones; pairs with high agreement but still close edit-distance are false positives from the bottom-up mining step and should be relabeled as non-fuzzy confusable pairs (still useful as a hard-boundary comparison set).

### 8.3 Speaker diversity and documentation

Must control/document per speaker and per clip: `speaker_id` (anonymized), `age_bracket`, `gender` (optional, self-reported), `dialect_region`, `recording_device`, `timestamp`. Preferred shape favors breadth of speakers per pair over breadth of pairs, specifically because cross-speaker variation is the mechanism expected to produce genuine acoustic overlap (as opposed to single-speaker idiosyncrasy).

### 8.4 Audio preprocessing, segmentation, alignment, QC pipeline (existing plan, retained)

1. Ingestion into `clips/<pair_id>/<speaker_id>_<word>.wav`.
2. Quality filtering: reject clips below minimum duration/SNR threshold, reject silence/noise-only clips, flag mismatched word/pair labels for manual review.
3. Normalization: resample to 16kHz mono, trim silence, loudness-normalize.
4. Deduplication.
5. **Speaker-disjoint** train/validation/test split (no speaker in more than one split — critical for RQ3/RQ5 to be valid, see Section 12).
6. Labeling: attach phoneme sequences, confusion_type, base-corpus frequency, **and the new fuzzy-annotation fields (8.2)**.
7. Base-corpus frequency alignment (weight confusability by real-world occurrence rate).
8. Export to a training-ready manifest (JSON/CSV per split).
9. **Forced alignment pass:** run a forced aligner over Layer 1 base-corpus utterances (and Layer 2 carrier-sentence clips, where present) to get phoneme-level time boundaries — required input for Section 9's fuzzy-boundary analysis.

### 8.5 Consent/licensing

Open item carried over from `datacollection.md`: finalize consent/licensing terms for community-submitted audio before any public contribution pipeline opens; anonymize speaker identity in all stored metadata by default.

---

## 9. Fuzzy Phoneme Boundaries (Not Assuming Discrete Boundaries)

Rather than assuming a forced-aligner's hard boundary timestamp is ground truth, this project treats phoneme boundaries as a **region of uncertainty**, operationalized as:

1. Run standard forced alignment to get an initial hard boundary estimate per phoneme.
2. Around each boundary, extract a symmetric window (e.g. ±30–50ms) and compute how classifier confidence (from Section 11's baseline classifier) changes across that window — a boundary is "fuzzy" if confidence stays near 0.5 across a wide window rather than sharply transitioning.
3. Cross-validate this against the human ambiguity ratings (Section 8.2) on carrier-sentence clips, where available — do listener judgments also waver in the same region?
4. Represent the boundary not as a single timestamp but as a **fuzzy interval** (e.g. a trapezoidal membership function: fully-phoneme-A, transition zone, fully-phoneme-B) or as a posterior probability curve over time (probabilistic alternative).

This directly operationalizes RQ1 and RQ2 at the sub-phoneme (temporal) level, complementing the pair-level fuzziness in Section 8.2.

---

## 10. Representing Uncertainty and Overlap Between Similar Phonemes

Two representational families to build and compare, not to pick a winner in advance:

### 10.1 Fuzzy-logic representation

- Membership functions (triangular/trapezoidal/Gaussian) over acoustic feature space for each phoneme in a confusable pair, following the Bora et al. and Jain et al. precedents in Pillar 1.
- Fuzzy classification rule: assign membership degrees to each candidate phoneme rather than a single label; the fuzzy pair's "confusability score" for a token is a function of the gap between the top two memberships (small gap = high fuzziness).

### 10.2 Probabilistic representation

- Posterior class probabilities from a probabilistic classifier (Section 12) as the uncertainty signal.
- **Gaussian Process Classification (GPC)** as the flagship probabilistic method (Section 13) — GPC gives a principled posterior *and* a separate epistemic-uncertainty estimate (how much is "the model doesn't know" vs. "the input is genuinely ambiguous"), which fuzzy-logic membership functions do not distinguish by construction. This distinction is itself a testable claim (RQ4).

### 10.3 Comparison axis

The core Layer 2/3 experimental question is not "fuzzy logic vs. probability in the abstract" but: **which representation correlates better with the human-annotated ambiguity ground truth from Section 8.2**, and which is more useful as an input feature to the downstream disambiguation classifier (Section 14/RQ5).

---

## 11. Baseline Experiments

Before any fuzzy/probabilistic method, establish conventional baselines:

1. **Hard-boundary phoneme classifier:** standard multiclass classifier (e.g. Random Forest per Breiman 2001, or a small feed-forward net) over acoustic features (Section 12), trained to output a single discrete phoneme label. This is the "crisp" comparison point for everything downstream.
2. **Off-the-shelf ASR substitution baseline:** run Whisper-Nepali / the CNN-Transformer LVCSR directly on Layer 2 clips (in carrier-sentence form) and measure raw substitution-error rate on the confusable pairs, with no fuzzy/probabilistic layer at all — this is the real-world "do nothing extra" baseline that Section 16's final evaluation must beat.
3. **Simple edit-distance/G2P baseline** for the word-level disambiguation task (no acoustic model at all — purely orthographic/phonetic string distance) — the cheapest possible baseline, useful for sanity-checking that acoustic modeling adds anything.

---

## 12. Acoustic Features to Extract

| Category | Features | Notes |
|---|---|---|
| Spectral | MFCCs (+ deltas/delta-deltas), formant frequencies (F1–F3, critical for vowel-length and retroflex/dental contrasts), spectral centroid/spread | Standard phoneme-classification features |
| Temporal | Voice Onset Time (VOT — critical for aspirated/unaspirated contrast), segment duration, zero-crossing rate | Directly targets the aspiration and vowel-length fuzzy contrasts in Section 3.2 |
| Prosodic | Pitch (F0) contour, energy/intensity contour, stress placement | Needed to check whether prosodic context modulates fuzziness (e.g. unstressed-position neutralization) |
| Learned representations | Self-supervised speech embeddings (Wav2Vec2 or Whisper encoder hidden states, ideally a Nepali/multilingual-fine-tuned variant) | For the neural-baseline comparison in Section 13.4; also usable as GPC kernel input features |

Feature extraction should be run once, cached, and versioned alongside the manifest — this both speeds up experimentation and keeps every downstream model (fuzzy, GPC, neural) trained on an identical feature set for fair comparison.

---

## 13. Models and Methods to Test

### 13.1 Fuzzy-logic approaches

- Fuzzy c-means clustering over acoustic feature space per confusable pair, to discover data-driven membership boundaries rather than hand-specified ones.
- Rule-based fuzzy inference (Mamdani-style), following the Bora et al. 2019 precedent, using VOT/formant/duration features as fuzzy-rule inputs.
- Fuzzy k-NN classification (each neighbor votes with a fuzzy membership weight).

### 13.2 Probabilistic approaches (non-GP)

- Naive Bayes / Gaussian Mixture Model per phoneme class, with class posteriors as the uncertainty signal.
- Bayesian logistic regression (for a Bayesian-but-simpler comparison point before GPC).
- Random Forest with predicted class-probability output (not just hard vote) — a cheap probabilistic baseline using the same base classifier family as Section 11's hard baseline, isolating the effect of using soft vs. hard output from the same model.

### 13.3 Gaussian Process Classification (flagship method)

- **Formulation:** binary GPC per confusable pair (phoneme A vs. phoneme B) as the primary setting, since most Nepali confusable contrasts in Section 3.2 are pairwise; extend to multiclass GPC (one-vs-rest or a softmax-based multiclass GP) if group confusions (e.g. श/ष/स) require it.
- **Kernels/covariance functions to test:** RBF (squared-exponential) as the default; Matérn (ν=3/2, 5/2) for less-smooth acoustic feature spaces; a combination/additive kernel over feature subgroups (spectral + temporal + prosodic) to test whether feature-group-aware kernels outperform a flat concatenated-feature RBF kernel.
- **Likelihood/approximate inference:** Laplace approximation as the first, cheapest approach; Expectation Propagation (EP) as a stronger alternative if Laplace underperforms; consider variational sparse GP (Hensman et al. 2015) if the dataset size makes exact/Laplace GPC too slow.
- **Uncertainty measures:** predictive posterior variance (epistemic uncertainty) as the primary output; predictive entropy of the class posterior as a secondary, more directly comparable-to-fuzzy-membership measure.
- **Why GPC specifically:** unlike a plain probabilistic classifier, GPC gives a *principled, non-parametric* posterior with an explicit uncertainty estimate that separates "few similar training examples seen" from "genuinely ambiguous acoustic input" — directly relevant to RQ4, and well-suited to Layer 2's necessarily small dataset (GPC is a strong choice precisely because it doesn't need huge data to give calibrated uncertainty, unlike deep neural approaches).

### 13.4 Neural / modern speech-representation baselines (for comparison, not the core contribution)

- Fine-tuned Wav2Vec2 or Whisper-encoder-based phoneme classifier (hard-decision) — the "what would a modern deep model do with no explicit uncertainty modeling" comparison.
- Same neural backbone with an added Monte Carlo Dropout or deep ensemble uncertainty estimate — the "modern deep-learning answer to uncertainty" comparison point against GPC, since this is the most likely reviewer question ("why not just use dropout uncertainty from a fine-tuned Wav2Vec2 model instead of GPC?").

---

## 14. Downstream Task — Confusable-Word Disambiguation

Following the existing paper blueprint's methodology: take the fuzzy/probabilistic/GPC phoneme-level uncertainty scores and feed them as features (alongside edit-distance and NepBERTa-derived context embeddings) into a word-level disambiguation classifier operating on ASR N-best/1-best output, to test RQ5. This is the link between the phoneme-level contribution (Layers 1–2) and the ASR-relevant contribution (Layer 3).

---

## 15. Experimental Design

### 15.1 Splitting and leakage prevention

- **Speaker-independent splits, mandatory:** no speaker's clips appear in more than one of train/validation/test, for both Layer 1 background-ASR use and Layer 2 fuzzy-pairs data.
- **Pair-level holdout as a secondary split:** in addition to speaker-independent splits, hold out an entire subset of confusable *pairs* (not just speakers) from training, to test whether fuzziness patterns generalize to unseen phoneme-pair types, not just unseen speakers of known pairs.
- No overlap between the pairs used to *seed* the bottom-up candidate mining (Section 4) and the pairs used for final held-out evaluation, to avoid the seed-list construction leaking into evaluation.

### 15.2 Ablations

- Feature-group ablation: spectral-only vs. +temporal vs. +prosodic vs. +learned embeddings, to identify which feature categories actually drive fuzzy-boundary detection accuracy.
- Representation ablation: fuzzy-logic-only vs. probabilistic-only vs. GPC vs. neural-uncertainty, isolating which uncertainty formalism contributes most.
- Kernel ablation within GPC: RBF vs. Matérn vs. combined kernel.
- Dataset-size ablation: performance vs. number of speakers per pair (does the "fewer pairs, more speakers" collection strategy from Section 8.1 actually pay off, or would more pairs with fewer speakers have worked as well?).
- Downstream ablation: word-level disambiguation with vs. without the fuzzy/GPC uncertainty feature, isolating RQ5 specifically.

### 15.3 Statistical testing

- Paired significance testing (e.g. McNemar's test for classifier comparisons on the same test items, or a paired bootstrap over test-set accuracy) between each candidate method and the hard-boundary baseline.
- Correlation testing (e.g. Spearman) between model uncertainty scores and human inter-annotator disagreement (Section 8.2), as the core calibration test for RQ4.
- Report confidence intervals, not just point estimates, given the necessarily modest dataset size (300–500+ clips) — effect sizes matter more than raw significance at this scale.

---

## 16. Evaluation Metrics

| Category | Metrics |
|---|---|
| Classification | Accuracy, precision/recall/F1 per phoneme class and per confusable pair, confusion matrices (crisp vs. fuzzy-aware) |
| Calibration | Expected Calibration Error (ECE), reliability diagrams, correlation between model uncertainty and human disagreement (Section 15.3) |
| Uncertainty quality | Predictive entropy / posterior variance vs. correctness (does high uncertainty predict errors?); epistemic vs. aleatoric decomposition where available (GPC) |
| Boundary-level | Agreement between fuzzy-boundary estimate (Section 9) and human-perceived ambiguity window |
| ASR-level (downstream) | Phoneme Error Rate (PER), Word Error Rate (WER)/Character Error Rate (CER) on confusable-word-containing utterances specifically (not overall corpus WER, which would dilute the effect) |
| Efficiency | Training/inference time per method, relevant given the low-resource/practical-deployment framing |

---

## 17. Visualization and Error Analysis

- Fuzzy membership curves / posterior probability curves plotted against the acoustic feature axis (e.g. VOT) for each confusable pair, to visually show where the "fuzzy zone" sits.
- Confusion matrices with cell shading by model confidence, not just raw counts.
- Reliability diagrams (calibration curves) per method.
- t-SNE/UMAP projection of acoustic feature space per confusable pair, colored by human ambiguity rating, to visually validate that "fuzzy" tokens cluster near the decision boundary rather than being scattered noise.
- Systematic error analysis: for every misclassified or high-uncertainty token, log dialect region, speaker demographic, and recording condition — to check whether fuzziness is evenly distributed or concentrated in specific dialects/conditions (a likely and important finding either way).

---

## 18. Expected Results and Alternative Outcomes

- **Expected (hypothesis-confirming) outcome:** GPC/probabilistic models show measurably higher accuracy on the fuzziest tokens (those with low human inter-annotator agreement) than the hard-boundary baseline, with uncertainty scores that correlate significantly with human disagreement, and the downstream word-disambiguation task shows a WER/CER improvement on confusable-word utterances specifically.
- **Plausible partial outcome:** Fuzzy/probabilistic methods improve calibration and interpretability (useful for error analysis and flagging low-confidence ASR output for human review) without improving raw classification accuracy over the hard baseline — still a genuine, reportable contribution, since a well-calibrated "I'm not sure" signal has real downstream value even without an accuracy gain.
- **Null/negative outcome:** No meaningful uncertainty-quality gain over a simple softmax-confidence baseline from the hard classifier — in this case the contribution shifts to a negative/cautionary result about GPC's practical value at this dataset scale for this task, still worth reporting, and the project should pivot toward diagnosing *why* (insufficient data for GPC, wrong kernel/features, or genuinely low fuzziness in the sampled pairs).
- **Data-driven surprise scenario:** bottom-up mining (Section 4) surfaces confusable pairs not predicted by phonological theory — this is itself a reportable finding about Nepali spoken-register confusability, independent of the modeling results.

---

## 19. Limitations and Threats to Validity

- Small, low-resource training data (300–500+ clips) limits statistical power, especially for the pair-level holdout ablation in Section 15.1.
- Fuzzy-membership functions may be hand-tuned rather than fully learned in the fuzzy-logic condition, introducing subjectivity that GPC avoids — flag this asymmetry explicitly rather than hiding it.
- Human ambiguity annotations (Section 8.2) are themselves noisy and depend on annotator pool size/expertise; report inter-annotator agreement as a limitation, not just a ground-truth input.
- Single ASR back-end evaluated for the downstream task (Section 14) — findings may not transfer to other ASR architectures.
- Dialectal coverage of collected speakers may be uneven (volunteer/community-contribution bias) — document and report, don't claim representativeness beyond what was actually collected.
- GPC scalability: exact/Laplace GPC does not scale gracefully past a few thousand points — if the dataset grows substantially, results at small vs. larger scale may not be directly comparable without switching to sparse/variational GPC, which itself changes the uncertainty estimates being compared.

---

## 20. Reproducibility Requirements

- Version-controlled `pairs.tsv` and `metadata.tsv` as the single source of truth (already the plan in `datacollection.md`), with clear versioning/tagging at each dataset snapshot used for a reported experiment.
- Fixed random seeds and documented train/val/test speaker assignment, checked into the repo (not regenerated ad hoc per run).
- All feature-extraction and preprocessing code checked into the repo with a single entry-point script per stage (ingestion → QC → normalization → split → feature extraction → training).
- Model configs (kernel choice, hyperparameters, likelihood approximation) logged per experiment run, not just the final reported numbers.
- A `requirements.txt`/environment lockfile so results are reproducible on a fresh environment.
- Clear separation between "exploratory" notebooks and the final reproducible pipeline scripts used for reported results.

---

## 21. Software, Libraries, and Implementation Structure

| Purpose | Tooling |
|---|---|
| Audio I/O, feature extraction | `librosa`, `torchaudio`, `praat-parselmouth` (for formants/VOT-style measurements) |
| Forced alignment | Montreal Forced Aligner, or a wav2vec2-CTC-based aligner if MFA lacks a Nepali acoustic model (likely — a Nepali-adapted alignment approach may need to be built, flag as an open item) |
| Fuzzy logic | `scikit-fuzzy` |
| Classical ML baselines | `scikit-learn` (Random Forest, Naive Bayes, Logistic Regression) |
| Gaussian Process Classification | `GPflow` or `GPyTorch` (both support Laplace/EP/variational GPC and multiple kernels; `GPyTorch` preferred if GPU-accelerated sparse GPC is needed at scale) |
| Neural baselines | `transformers` (Wav2Vec2/Whisper), `PyTorch` |
| Statistical testing | `scipy.stats`, `statsmodels` |
| Visualization | `matplotlib`/`seaborn` for static plots, `umap-learn`/`scikit-learn` t-SNE for embedding visualization |
| Experiment tracking | lightweight (e.g. a logged CSV/JSON per run, or Weights & Biases if available) — must support the reproducibility requirements in Section 20 |

**Suggested repo structure** (extending the current `Proposal/` / `Papers/` layout):

```
data/            # pairs.tsv, metadata.tsv, manifests (not raw audio if large — document storage location)
clips/           # <pair_id>/<speaker_id>_<word>.wav
src/
  g2p/           # rule-based Devanagari-to-phoneme converter
  preprocessing/ # ingestion, QC, normalization, splitting
  features/      # spectral/temporal/prosodic/learned-representation extraction
  models/
    fuzzy/
    probabilistic/
    gpc/
    neural/
  eval/          # metrics, calibration, statistical tests
  viz/
notebooks/       # exploratory only, not the reproducible path
```

---

## 22. Experimental Timeline

Grounded in the existing 14-day cap noted in the compiled bibliography for the *initial* pilot/proposal phase; this proposal extends that into a realistic full-research timeline for the three-layer structure. Adjust to actual course/thesis deadlines.

| Phase | Duration | Deliverable |
|---|---|---|
| **Layer 1 — Foundational** | Weeks 1–2 | Nepali fuzzy phoneme inventory (Section 3.3), completed literature review incl. new Pillar 5 (GP/uncertainty), G2P converter v1, seed candidate confusable-pair list |
| **Layer 2 — Experimental (data)** | Weeks 3–5 | Layer 2 dataset collected (300–500+ clips), QC pipeline run, fuzzy-annotation layer (human ambiguity ratings) collected, speaker-disjoint splits finalized |
| **Layer 2 — Experimental (baselines + fuzzy)** | Weeks 5–7 | Baselines (Section 11) implemented and evaluated; fuzzy-logic methods (13.1) implemented and evaluated |
| **Layer 3 — Advanced (probabilistic/GPC)** | Weeks 7–9 | Probabilistic methods (13.2) and GPC (13.3) implemented, kernel/likelihood ablations run, calibration analysis (RQ4) completed |
| **Layer 3 — Advanced (neural comparison + downstream)** | Weeks 9–10 | Neural baselines (13.4) run; downstream disambiguation task (Section 14/RQ5) evaluated |
| **Analysis & writing** | Weeks 10–12 | Statistical testing, visualizations, error analysis, full write-up per the existing paper blueprint (~5,000–6,500 words target) |
| **Review & submission prep** | Weeks 12–13 | Internal review, reference verification (resolve flagged/unverified citations), figure/table polish |
| **Buffer** | Week 14 | Contingency for data-collection shortfall or re-running failed ablations |

---

## 23. Path From Exploratory Study to Publication-Level Contribution

1. **Minimum viable version (course/coursework level):** Layer 1 inventory + Layer 2 dataset (even at reduced scale, e.g. 10–15 pairs) + baseline vs. one fuzzy method vs. GPC comparison, evaluated with basic accuracy/F1 and a calibration check. This alone satisfies the existing paper blueprint's target length and structure.
2. **Strong workshop/short-paper contribution:** add the full ablation suite (Section 15.2), the downstream disambiguation task (RQ5), and rigorous statistical testing — this is what elevates the work from "a classifier was built" to "a hypothesis was tested with evidence."
3. **Full conference/journal-level contribution:** extend dataset scale and dialectal coverage, add the neural-uncertainty comparison (13.4) as a genuinely strong baseline (not a token one), release the fuzzy-pairs dataset and G2P converter publicly as a reusable resource for Nepali/Indic speech research (this is a citable artifact contribution independent of the modeling results), and generalize the framework's applicability discussion to other low-resource languages.

---

## 24. Applications

- **Nepali ASR:** post-processing/re-ranking signal for confusable-word disambiguation, or an uncertainty-aware decoding feature.
- **Pronunciation modeling / CALL (computer-assisted language learning):** flagging genuinely ambiguous pronunciation targets for Nepali learners, rather than treating all phoneme errors as equally "wrong."
- **Speech technology broadly:** a template for uncertainty-aware phoneme modeling applicable to other low-resource, phonologically dense languages (other Indic languages, tonal languages with gradient contrasts).
- **Low-resource language technology:** the released fuzzy-pairs dataset, G2P converter, and annotation methodology are reusable artifacts even independent of the GPC modeling results.
- **Accessibility and health-voice pipelines:** reducing meaning-altering ASR errors (e.g. confusable medical/health terms) in downstream Nepali voice applications such as the previously noted Swasthya Swar-style pipeline.

---

## 25. Open Items to Resolve Before Full Drafting

Carried over and extended from the existing blueprint's open items:

1. Confirm final dataset composition (SLR54 + in-house Layer 2 only, or also NepBERTa corpus for text-mining context features).
2. Decide GPC library (`GPflow` vs. `GPyTorch`) based on available compute (GPU access affects this significantly for sparse GP at scale).
3. Confirm forced-alignment tooling for Nepali — no off-the-shelf Nepali acoustic model may exist for MFA; may need a wav2vec2-CTC-based aligner instead, which changes the Section 9 boundary-fuzziness pipeline.
4. Decide human-annotation pool size/recruitment method for the ambiguity ratings in Section 8.2 (native-speaker peers, paid crowdsourcing, or a small fixed panel) — this materially affects Section 15.3's calibration analysis validity.
5. Resolve the flagged/unverified citations (Levenshtein 1966 DOI; several "author list to confirm" entries) before final submission.
6. Confirm consent/licensing terms for community-submitted audio before opening any public contribution pipeline.
7. Add the new Pillar 5 (GP/uncertainty methods) citations per Section 5.2 to the bibliography file.
