# Recording checklist

Practical checklist for producing clips for `Proposal/clips/`. Words and
filenames per pair are in [wordlist.md](wordlist.md); pair_id → phoneme
mapping and naming convention are in [README.md](README.md). This file
covers the recording session itself.

## Before you start

- [ ] **Room:** small, quiet, soft-furnished room (carpet/curtains/sofa beats
      a bare tiled room — less echo). Turn off fans/AC, close windows.
- [ ] **Device:** phone voice memo app is fine. If you have a headset/lapel
      mic, use it — consistent mic-to-mouth distance matters more than mic
      quality.
- [ ] **Distance:** keep the mic ~15–20cm from your mouth, same distance for
      every clip in a session. Don't let it drift closer/further mid-session.
- [ ] **Format:** record at the highest quality your device allows (WAV if
      possible; otherwise export/convert to WAV before saving into
      `clips/`). Don't worry about matching 16kHz yourself — the pipeline
      resamples on ingest.
- [ ] **Silence buffer:** leave ~1s of silence before and after each word —
      makes trimming cleaner and avoids clipping the onset (important for
      the VOT-focused pairs: `ba_bha`, `ka_kha`).

## Per-word recording

- [ ] One word per clip, not a whole list read continuously — split/trim
      before saving, or record each word as its own take.
- [ ] Read at a natural, steady pace — not slowed-down "careful" speech,
      not rushed either. Natural conversational tempo is what the ASR/
      classifier pipeline will see in real use.
- [ ] Record each word **2–3 times** (per wordlist.md's note) and keep the
      cleanest take — no stumbles, no background noise spike, no clipped
      onset/offset.
- [ ] For the retroflex/dental and sibilant pairs especially: don't
      exaggerate the contrast to make it "clearer" — natural pronunciation
      is the point, including any merger tendency (e.g. ष/स).

## Multi-speaker sessions (priority: more `ba_bha` speakers first)

- [ ] Each new speaker gets their own `speaker_id` tag (short, anonymized —
      e.g. `spk02`, `spk03`, not a real name) used consistently in every
      filename: `<speaker_id>_<word>_<phoneme>.wav`.
- [ ] Brief the speaker on natural pace before recording — don't have them
      read the Devanagari cold if they're not a fluent reader; say the word
      aloud once yourself first if needed, off-mic.
- [ ] Same room/device/distance setup as above, ideally the same session
      setup used for the existing `ba_bha` clips for consistency.

## Before handing off

- [ ] Filenames follow the convention exactly: `<speaker_id>_<word>_<label>.wav`,
      lowercase, no spaces — check against wordlist.md.
- [ ] No duplicate filenames within a pair folder.
- [ ] Quick listen-back: no silence-only files, no cut-off words, no
      overlapping background speech/noise.
- [ ] Drop files into the matching `clips/<pair_id>/` folder (or hand them
      over however's easiest — folder path, zip, or paste into chat) and
      flag which pairs/speakers were added so the pipeline re-run targets
      the right thing.

## Priority order

1. 2–3 more speakers on the existing `ba_bha` words (biggest signal gain —
   current 0.30 baseline number is from 1 speaker, 10 clips).
2. First pass (1 speaker, 5 words/side) on the 7 remaining pairs: `ka_kha`,
   `tta_ta`, `dda_da`, `i_ii`, `u_uu`, `sha_sa`, `ssha_sa`.
3. Additional speakers on those 7 once a first pass exists for all of them.
