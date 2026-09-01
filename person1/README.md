# Person 1 — Heavy Processing (Audio, ASR, Similar-Sounding Words)

**Role:** all GPU / bulk-CPU / large-download work. Produces the linguistic
building blocks (word + phoneme database, similar-sounding-word lookup) and the
real ASR error data that Persons 2 and 3 consume. Everything below the ASR
stays lightweight; all heavy compute lives here.

**Environment:** conda env `asr`
(`C:\Users\Hp\miniconda3\envs\asr\python.exe`). Run with `PYTHONUTF8=1`.

---

## What Person 1 does (steps → scripts)

| # | Step | Script / artifact |
|---|------|-------------------|
| P1.1 | Parse SLR54 `utt_spk_text.tsv` → vocab, frequencies, speaker/utterance stats; speaker-disjoint train/val/test splits | `scripts/word_stats.py` (done) |
| P1.2 | Nepali Unicode normalization (NFC), ZWJ/ZWNJ strip | inside `word_stats.py` |
| P1.3 | Devanagari tokenization (danda/punct stripped) | inside `word_stats.py` |
| P1.4 | G2P Devanagari → IPA (Piper/espeak-ng `ne`); affricate normalization (`t͡s`/`d͡ʒ`/`t͡ʃ`); validate on manual sample | `scripts/phonemize.py` + `scripts/ipa_utils.py` (done) |
| P1.5 | PanPhon: IPA segments → numeric articulatory feature vectors | `scripts/phoneme_db.py` (done) |
| P1.6 | Phonetic feature edit distance between words (vectorized DP, validated vs panphon) | `scripts/phonetic_distance.py` (done) |
| P1.7 | Word+phoneme DB: freq, speakers, all occurrence IDs, source; length×onset buckets | `scripts/phoneme_db.py` (done) |
| P1.8 | Similar-sounding lookup: same-onset length L±1 candidates, exact PanPhon distance, top-k | `scripts/similar_words.py` (done) |
| P1.9 | **HEAVY** — download ~5–15 h SLR54 audio subset; run pretrained Nepali ASR (Indic wav2vec-2.0 / Vakyansh) → hypotheses + confidence | `scripts/download_slr54.py` (written), ASR step TBD |
| P1.10 | Word-level alignment ref vs ASR hyp; WER; confusion-prior estimation | TBD script |
| P1.11 | Motivation stats (similar-sounding substitutions as dominant error class) | TBD |

## What Person 1 produces (final outputs)

- `db/phoneme_db.tsv` + `db/phoneme_db.pkl` — word, source, IPA, segments,
  per-segment PanPhon features, freq, n_utts, n_speakers, all occurrence file_ids
- `db/phoneme_occurrences.tsv` — word → all OpenSLR utterance IDs
- `artifacts/similar_words.tsv` — query word × rank × similar word × distance
  × IPA × freq × n_utts
- `asr_error_corpus.tsv` — occurrence × reference_word × ASR_hypothesis × confidence
- `confusion_priors.tsv` — substitution/confusion priors from the ASR pass
- Pronunciation lexicon (word → IPA), trained LDA model (optional, shared with P2)

## Handoff to Person 2

- `similar_words.tsv` / `db/phoneme_db.tsv` (→ P2.5 features: similar-word
  neighbors, phoneme features, distance)
- `asr_error_corpus.tsv` (→ P2.3 masking, P2.5 feature f5 ASR confidence)
- `confusion_priors.tsv` (→ P2.5 feature f6)
- Speaker-disjoint split definition (→ P3.2)
- Train/val/test occurrence boundaries

## Stats to save for the report (tables)

- Corpus summary: utterances, speakers, tokens, vocabulary size
- G2P validation agreement on the manual sample
- PanPhon distance calibration histogram / threshold sensitivity
- Similar-word coverage (share of words with ≥1 similar word), distance-by-rank
- WER of the ASR pass; confusion-prior counts

## Figures to generate for the report

- Vocabulary frequency distribution (log-log Zipf plot)
- Similar-word distance histogram + rank-vs-distance plot + coverage bar
- Heatmap of PanPhon feature-edit distance for a sample of top similar words
- Confusion-prior matrix (top substituted pairs)

## Status

- [x] P1.1–P1.8 implemented; full-vocabulary pipeline validated (48,995 words)
- [x] Downloader written; **full audio subset + ASR pass (P1.9) not yet run**
- [ ] P1.10 alignment/confusion-prior script, P1.11 motivation stats, report
  figures, EDA run










  $env:PYTHONUTF8='1'
python person1\scripts\word_stats.py --index person1\data\slr54\utt_spk_text.tsv --out-dir person1\artifacts
python person1\scripts\phonemize.py --words person1\artifacts\vocab.tsv --out person1\artifacts\phonemes.tsv
python person1\scripts\phoneme_db.py --vocab person1\artifacts\vocab.tsv --phonemes person1\artifacts\phonemes.tsv --index person1\data\slr54\utt_spk_text.tsv --db-dir person1\db
python person1\scripts\similar_words.py --dump --db person1\db\phoneme_db.pkl --out person1\artifacts\similar_words.tsv
python person1\scripts\eda\run_all.py --skip-pipeline