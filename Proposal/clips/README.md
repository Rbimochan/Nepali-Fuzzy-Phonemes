# clips/ — pilot data collection

Scaffolding for the 8 pilot confusable pairs from [bimochan.html](../bimochan.html) §3. One subfolder per `pair_id`.

## pair_id → phoneme pair

| pair_id | Pair | Contrast type |
|---|---|---|
| `ba_bha` | ब / भ | Aspirated vs unaspirated |
| `ka_kha` | क / ख | Aspirated vs unaspirated |
| `tta_ta` | ट / त | Retroflex vs dental |
| `dda_da` | ड / द | Retroflex vs dental |
| `i_ii` | इ / ई | Vowel length |
| `u_uu` | उ / ऊ | Vowel length |
| `sha_sa` | श / स | Sibilant merger |
| `ssha_sa` | ष / स | Sibilant merger |

## File naming convention

```
clips/<pair_id>/<speaker_id>_<word>_<phoneme>.wav
```

- `speaker_id`: short anonymized tag, e.g. `spk01`, `spk02`
- `word`: the Devanagari word the clip isolates (or a romanized slug), e.g. `बानी`
- `phoneme`: which member of the pair this clip is a token of, e.g. `ba` or `bha`

Example: `clips/ba_bha/spk01_baani_ba.wav`

## Recording checklist (per §2 of bimochan.html)

- [ ] 16kHz mono, minimal background noise, trimmed silence
- [ ] Each pair: aim for a handful of words per phoneme per speaker, not just one token
- [ ] At least 2–3 speakers per pilot pair if self-recording (per open question in §8 — leaning toward recording now rather than waiting on the community pipeline)
- [ ] No duplicate filenames within a pair folder

## Status

Empty — no clips collected yet as of 2026-08-19. Next: check whether any community-collected clips already exist for these 8 pairs before recording a self-collected pilot set.
