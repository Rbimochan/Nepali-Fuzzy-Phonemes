# Lane 1 Implementation Spec — Utsab (Data & Evidence Foundation)

Implementation spec for §4.2–4.3.2 of the locked proposal
(`Fuzzy_Phonemes_Nepali_ASR_Disambiguation.pdf`, March 2026), matching the
format of `Proposal/GP_Classification_Implementation.md`. This documents
what Lane 1 needs to produce and cross-references it against what already
exists in `person1/` — that folder predates the locked proposal (it was
built under an earlier P1/P2/P3 scheme, not this paper's Lane 1/2/3 split)
so there's real overlap but also a real gap. Written by the Lane 3 person
as a documentation pass, not an implementation — Utsab owns the actual
build.

---

## 1. What Lane 1 needs to produce (§4.2, Figure 3)

Per Figure 3's "Data and evidence foundation pipeline," six stages:

1. **Nepali Speech Data** — OpenSLR SLR54 (+ SLR43 if needed), audio +
   verified transcripts.
2. **Vocabulary & Occurrence Selection** — extract unique words, frequency
   analysis, select represented words.
3. **Symbolic Phonetic Evidence** — G2P → IPA, PanPhon features
   (**22-dimensional** per the proposal), phonetic distance ranking →
   confusable clusters (example given: `C001 [कल, खल, कला]`).
4. **ASR & Alignment** — Whisper ASR, reference-vs-hypothesis alignment,
   substitution detection, word boundary extraction.
5. **Audio-Derived Acoustic Evidence** — general features (MFCC, F0/energy,
   spectral, duration) **and** the four explicit phonetic cues:
   - **VOT** — waveform + spectrogram analysis, Praat
   - **F0 perturbation at vowel onset** — Praat pitch tracking
   - **Spectral tilt (H1-H2)** — LPC spectrum analysis
   - **Aspiration duration** — spectral prominence in high-frequency bands
   - plus temporal cues: closure duration, vowel duration
6. **Validated Handoff Package** — structured output to Sushank's Lane 2:
   confusable_word_clusters, openslr_occurrences, asr_results,
   acoustic_features, phonetic_cue_features.

## 2. Cross-reference against existing `person1/` work

`person1/` was built under an earlier, different task breakdown (P1–P11 in
`person1/README.md`) before this paper's Lane 1/2/3 structure was locked.
Mapping what it already covers against the spec above:

| Locked-proposal need | `person1/` status | Notes |
|---|---|---|
| Vocabulary/occurrence selection | ✅ done (`scripts/word_stats.py`) | Speaker-disjoint splits already built. |
| G2P → IPA | ✅ done (`scripts/phonemize.py`, `scripts/ipa_utils.py`) | |
| PanPhon features | ⚠️ done, but **dimension mismatch** | `person1` uses PanPhon's `spe+` set, **24 features** (`db/README.md`); the locked proposal specifies **22-dimensional** PanPhon features. Needs a decision: which 22 (drop 2 from `spe+`), or update the proposal's number to 24 — not something to silently reconcile without asking. |
| Phonetic distance / confusable clusters | ✅ done (`scripts/phonetic_distance.py`, `scripts/similar_words.py`, `scripts/confusable_clusters.py`) | This is the strongest overlap — `similar_words.tsv` and `confusable_clusters.tsv` already exist as artifacts. |
| Whisper ASR + alignment | ⚠️ partially done | `person1/asr/run_whisper.py`, `build_audio_text_ipa.py`, `build_index.py` exist and have been run (`whisper_output.tsv`, `whisper_words.tsv`, `audio_text_ipa.tsv` all present as artifacts) — but this was for the earlier P1.9 "confusion prior estimation" goal, not explicitly validated against this proposal's "substitution detection + word boundary extraction" framing. Likely close, needs a compatibility check rather than a rebuild. |
| General acoustic features (MFCC, F0, energy, spectral, duration) | ❌ not found | No script producing these 8 general features (§4.3.1 of the GP-classification spec) currently exists in `person1/`. |
| **Explicit phonetic cues (VOT, F0 perturbation, H1-H2, aspiration duration)** | ❌ not built | Confirmed in `person1/README.md`: P1.10 (alignment/confusion-prior) and P1.11 (motivation stats) are marked `TODO`, and there's no Praat-based cue-extraction script anywhere in `person1/scripts/`. **This is the single biggest blocker for both Lane 2 (Sushank) and Lane 3 (Bimochan)** — see `Proposal/GP_Classification_Implementation.md` §7, which is blocked on exactly this. |
| Validated Handoff Package (structured output) | ❌ not built | The individual artifacts exist but aren't assembled into the single structured package Figure 3 describes as the Lane 2 handoff. |

## 3. What's actually left to build

In priority order (highest-impact first, since everything downstream is
blocked on this lane per the group's own Aug-20 discussion — "everyone
downstream is blocked on his output being correct"):

1. **VOT extraction** — waveform/spectrogram analysis via Praat, per word
   occurrence. This is the most load-bearing single feature (it's the
   *primary* cue in Figure 4's phonetic-cue branch) and the one the
   locked proposal's phonology examples center on (बाघ/भाग aspiration
   contrast).
2. **F0 perturbation, spectral tilt (H1-H2), aspiration duration** — same
   Praat-based approach, can likely share tooling/setup with VOT
   extraction once that pipeline exists.
3. **General acoustic features** (§4.3.1's 8 features via `librosa`) — the
   most mechanical of the remaining work; no novel design needed, just
   needs to run per occurrence and land in the handoff package.
4. **Resolve the PanPhon dimension mismatch** (22 vs 24) with the team —
   low effort, but silently picking one would misrepresent either the
   code or the paper.
5. **Assemble the Validated Handoff Package** — package the above plus the
   already-done G2P/PanPhon/cluster/ASR outputs into the single structured
   format Figure 3 promises to Lane 2.
6. **Confirm the ASR/alignment output is compatible** with what this
   proposal's Lane 2/3 actually need (substitution detection, word
   boundaries) rather than assuming the earlier P1.9-era work transfers
   as-is.

## 4. Handoff contract (what Lanes 2 and 3 are waiting on)

Per `Proposal/GP_Classification_Implementation.md` §1, Lane 3's evidence
table expects these columns to trace back to this lane's output:
`VOT`, `F0_pert`, `H1H2`, `AspDur` (all four blocked per §2 above), plus
`occurrence_id` and `candidate_word` linking back to the ASR
hypothesis/speech segment. Lane 2 additionally needs the confusable
clusters and general acoustic features to compute GMM/DTW scores.

Until items 1–3 above land, Lane 2 and Lane 3 can only build and test
against synthetic placeholder data (as `gp_classifier_gpy.py` and
`ablation_grid.py` currently do) — real numbers can't exist before real
phonetic cues do.
