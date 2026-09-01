# Person 1 — Phoneme Database (`db/`)

The primary Person-1 deliverable: a word + phoneme database built from the
SLR54 corpus. Everything else (similar-word lookup, Person 2/3 features) reads
from here.

## Files

| File | Contents |
|------|----------|
| `phoneme_db.pkl` | Machine-readable DB: entries, lookup buckets, metadata. Load with `pickle`, or via `SimilarityIndex` in `scripts/similar_words.py`. |
| `phoneme_db.tsv` | Human-readable table: `word, source, ipa, segments, n_segments, freq, n_utts, n_speakers` |
| `phoneme_occurrences.tsv` | `word, source, n_utts, file_ids` — every OpenSLR utterance ID where each word occurs (comma-separated) |

## Schema (phoneme_db.pkl)

```
{
  "feature_set": "spe+",        # PanPhon feature set (24 features)
  "feature_names": [...],       # 24 names, e.g. syl, son, cons, ...
  "source": "slr54",            # corpus source; extensible to other corpora
  "entries": {
     "<source>::<word>": {
        "word": <str>,          # actual Devanagari word
        "source": <str>,        # "slr54"
        "ipa": <str>,           # pre-normalized IPA (affricates, ZWJ-stripped)
        "ipa_orig": <str>,      # raw G2P output
        "segs": [<str>, ...],   # phonological segments
        "feats": [[24 floats] per segment],  # numeric articulatory features
        "freq": <int>,          # token frequency in the corpus
        "n_utts": <int>,        # utterances containing the word
        "n_speakers": <int>,    # speakers using the word
        "file_ids": [<str>, ...] # all OpenSLR utterance IDs (occurrence index)
     }, ...
  },
  "word_index": { <word>: [(<source>, entry), ...] },
  "buckets": {
     "<source>::<n_segments>::<first_segment>": [<word>, ...]
  }
}
```

- Segment features use panphon's `spe+` set (24 features, values -1/0/1).
  Converted to floats by `scripts/phonetic_distance.py` (`to_numeric`).
- Affricates are normalized to breved forms (`t͡s`/`d͡ʒ`/`t͡ʃ`) via
  `scripts/ipa_utils.py`; ZWJ/ZWNJ are stripped in `scripts/word_stats.py`.
- The bucketing key `(source, n_segments, first_segment)` powers the fast
  similar-word lookup in `scripts/similar_words.py`.

## Rebuild

```
python person1\scripts\phoneme_db.py \
  --vocab person1\artifacts\vocab.tsv \
  --phonemes person1\artifacts\phonemes.tsv \
  --index person1\data\slr54\utt_spk_text.tsv \
  --db-dir person1\db
```