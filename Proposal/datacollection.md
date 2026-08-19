# Data Collection & Processing Plan

Fuzzy Phonemes: Classification, Optimization, and Text Mining for Disambiguating
Confusable Words in Low-Resource Nepali ASR

## 1. Overview

No existing dataset labels confusable/similar-sounding Nepali word pairs. This
project builds one from scratch, layered on top of existing raw Nepali speech
corpora. Two layers:

- **Layer 1 — Base corpus**: existing public Nepali speech/text resources,
  used for vocabulary extraction, frequency statistics, and background ASR.
- **Layer 2 — Confusable-pairs dataset**: newly collected, purpose-built
  minimal-pair audio + metadata, the actual contribution of this project.

## 2. Layer 1 — Base Corpus

| Source | Content | Size | Use |
|---|---|---|---|
| OpenSLR SLR54 | Nepali ASR transcribed speech (`utt_spk_text.tsv` + audio) | ~157k utterances, ~9.3GB | Vocabulary/frequency source, background ASR fine-tuning, substitution-error mining |
| OpenSLR SLR43 | Nepali TTS, female speakers, `line_index.tsv` + audio | ~800MB | Acoustic diversity, additional clean transcriptions |
| Mozilla Common Voice (Nepali) | Crowd-sourced speech + validated transcripts | growing, variable quality | Speaker/accent diversity |

Neither SLR54 nor SLR43 includes a pronunciation lexicon — none exists for
Nepali. A rule-based Devanagari-to-phoneme (G2P) converter will be built
in-house (Section 4) rather than assumed to exist.

## 3. Layer 2 — Confusable-Pairs Dataset (New Collection)

### 3.1 Unit of collection

The atomic unit is a **pair**, not a single clip:

- `pair_id`
- `word_A`, `word_B` (Devanagari script)
- `phoneme_A`, `phoneme_B` (from G2P, Section 4)
- `confusion_type`: one of
  - minimal-phoneme (e.g. retroflex/dental, aspirated/unaspirated, vowel length)
  - orthographic/visual (similar spelling, different sound)
  - frequency-driven ASR bias (common word substituted for rare word)
- `source`: `phonetic-distance-candidate` | `manual` | `community-submitted` | `asr-error-mined`

### 3.2 Audio collection per pair

- Clip of `word_A` spoken in isolation
- Clip of `word_B` spoken in isolation
- (optional, encouraged) both words spoken inside a short carrier sentence,
  to capture in-context confusability separately from isolated-word confusability
- Metadata per clip: `speaker_id` (anonymized), `age_bracket`, `gender`
  (optional, self-reported), `dialect_region`, `recording_device`,
  `timestamp`

### 3.3 Target volume and shape

Initial collection target: 300–500 clips.

Preferred shape: **fewer pairs, more speakers per pair**, not many pairs from
one speaker. Speaker variation stresses the confusability signal more than
raw pair count.

- ~50 pairs × 2 words × 5–8 speakers ≈ 500–800 clips
- rather than ~250 pairs × 2 words × 1 speaker

### 3.4 Seeding the pair list

1. Extract vocabulary from SLR54 transcriptions.
2. Run rule-based Devanagari G2P (Section 4) to get phoneme sequences.
3. Compute phonetic edit-distance over all word pairs; keep close matches
   as candidates.
4. Manual filtering by native speaker(s): discard candidates that are
   edit-distance-close but not actually confusable in practice.
5. Open the filtered list to community contribution/expansion.

## 4. Grapheme-to-Phoneme (G2P) Conversion

Devanagari is largely phonemic, so a rule-based converter is feasible
without a pre-built lexicon. Rules must handle:

- Inherent vowel (schwa) deletion at word-final and certain medial positions
- Conjunct consonants (संयुक्ताक्षर)
- Aspirated vs unaspirated consonant pairs
- Retroflex vs dental consonant pairs
- Vowel length distinctions
- Nasalization (chandrabindu / anusvara)

Output: a phoneme sequence per word, in a consistent inventory (IPA or a
custom phoneme set), stored alongside each word in the pair list.

## 5. Contribution Pipeline

Goal: let other contributors submit clips without direct repo access friction.

- `pairs.tsv` — canonical, version-controlled list:
  `pair_id, word_A, word_B, phoneme_A, phoneme_B, confusion_type, source`
- `clips/<pair_id>/<speaker_id>_<word>.wav` — audio storage convention
- `metadata.tsv` — one row per clip, keyed by `pair_id` + `speaker_id`
- Contribution flow:
  1. Contributor picks an existing `pair_id` (or proposes a new pair via PR
     to `pairs.tsv`)
  2. Records `word_A` and `word_B` in isolation (phone recording acceptable)
  3. Submits clips + fills metadata via form or direct PR
  4. Submission reviewed for audio quality and correct pairing before merge
- All contributions merged via git, keeping `pairs.tsv` as the single source
  of truth and audio as append-only additions.

## 6. Processing Pipeline (Before Training)

1. **Ingestion**: collect raw clips into `clips/` per naming convention above.
2. **Quality filtering**:
   - Reject clips below a minimum duration/SNR threshold
   - Reject clips with no detected speech (silence/noise only)
   - Flag mismatched word/pair labels for manual review
3. **Normalization**:
   - Resample all audio to a single target sample rate (e.g. 16kHz mono)
   - Trim leading/trailing silence
   - Loudness normalization
4. **Deduplication**: detect and remove duplicate clips (same speaker,
   same word, identical recording resubmitted).
5. **Splitting**: partition into train/validation/test by **speaker**, not
   by clip, so no speaker appears in more than one split (avoids leakage).
6. **Labeling**: attach `phoneme_A`/`phoneme_B`, `confusion_type`, and
   base-corpus frequency statistics to each pair for downstream
   classification/optimization tasks.
7. **Base corpus alignment**: cross-reference pair vocabulary against
   SLR54/SLR43/Common Voice word frequencies, to weight confusability by
   real-world occurrence rate.
8. **Export**: final training-ready format (e.g. manifest JSON/CSV per
   split, referencing audio paths, phoneme sequences, and confusion labels).

## 7. Open Items

- Finalize phoneme inventory (IPA vs custom ARPAbet-style set for Nepali)
- Decide consent/licensing terms for community-submitted audio
- Define minimum quality bar (device, environment) for accepted clips
- Decide whether carrier-sentence recordings are required or optional at
  launch
