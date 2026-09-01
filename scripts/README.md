# OpenSLR SLR54 — Nepali ASR dataset download scripts

Scripts to download and verify the **Large Nepali ASR training data set**
([OpenSLR SLR54](https://www.openslr.org/54)).

Contents:
- `utt_spk_text.tsv` — index: `FileID \t anonymized UserID \t transcript` (~10.9 MB)
- `asr_nepali_0.zip` … `asr_nepali_f.zip` — 16 audio shards, ~589 MB each (~9.4 GB total)

## Quick start

```powershell
# full corpus (all 16 shards) into ../data/slr54
python scripts/download_slr54.py --out-dir data/slr54

# small subset for the Person-1 offline ASR pass (~2 shards)
python scripts/download_slr54.py --out-dir data/slr54 --max-shards 2 --extract

# faster with 8 parallel connections
python scripts/download_slr54.py --out-dir data/slr54 --jobs 8 --extract
```

## Verify

```powershell
python scripts/verify_slr54.py --data-dir data/slr54
python scripts/verify_slr54.py --data-dir data/slr54 --list-zips
```

## Notes

- Downloads are **resumable**: a `.part` file is kept per in-flight shard and
  restarted from the byte offset. If a final file exists and matches the
  server `Content-Length`, it is skipped.
- **Mirror fallback**: default primary mirror is `openslr`; on failure each
  file automatically tries `trmal`, `elda`, `magicdata`, then `us`
  (`us.openslr.org` occasionally has TLS problems; use `--insecure` only if
  you need it).
- Use `--max-shards N` to download only the first N shards — recommended for
  the ~5–15 h audio subset used in the ASR pass (plan step P1.8).
- `--extract` unpacks zips in place; `--delete-zips` also removes the archives.
- Shard zips may already contain their own copy of `utt_spk_text.tsv`;
  the standalone index file is fetched separately unless `--no-index`.
- Requires **Python 3.6+**. `tqdm` is recommended for progress bars
  (`pip install tqdm`); the script falls back to plain output if it is absent.
  No other packages are needed.

## Expected directory layout after download + extract

```
data/slr54/
├── utt_spk_text.tsv
├── asr_nepali_0.zip ... asr_nepali_f.zip
└── (after --extract) asr_nepali_<shard>/...
       ├── utt_spk_text.tsv
       └── data/...wav
```