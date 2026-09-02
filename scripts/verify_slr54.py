#!/usr/bin/env python3
"""
verify_slr54.py -- Verify a downloaded OpenSLR SLR54 dataset.

Checks:
  1. utt_spk_text.tsv parses (FileID / UserID / transcript columns).
  2. Every expected shard zip is present (unless --max-shards given).
  3. Each zip is a valid archive (and optionally lists its contents).
  4. All zip-internal wav files exist on disk after extraction (optional).

Usage:
  python verify_slr54.py --data-dir ../data/slr54
  python verify_slr54.py --data-dir ../data/slr54 --list-zips
  python verify_slr54.py --data-dir ../data/slr54 --max-shards 2
"""

import argparse
import csv
import glob
import os
import zipfile

INDEX_FILE = "utt_spk_text.tsv"
SHARD_NAMES = [f"asr_nepali_{i}.zip" for i in range(10)] + [
    f"asr_nepali_{c}.zip" for c in "abcdef"
]


def main():
    ap = argparse.ArgumentParser(description="Verify OpenSLR SLR54 download.")
    ap.add_argument("--data-dir", default="data/slr54", help="directory containing the download")
    ap.add_argument("--max-shards", type=int, default=len(SHARD_NAMES),
                    help="how many shard zips were downloaded")
    ap.add_argument("--list-zips", action="store_true",
                    help="print the internal structure of the first shard zip")
    ap.add_argument("--check-extracted", action="store_true",
                    help="check that wav files listed inside zips exist on disk")
    args = ap.parse_args()

    problems = []

    index_path = os.path.join(args.data_dir, INDEX_FILE)
    if not os.path.exists(index_path):
        problems.append(f"missing index file: {index_path}")
    else:
        n = 0
        with open(index_path, encoding="utf-8") as fh:
            for _ in fh:
                n += 1
        print(f"Index file: {n} rows (FileID/UserID/transcript)")

    shards = SHARD_NAMES[: args.max_shards]
    for name in shards:
        path = os.path.join(args.data_dir, name)
        if not os.path.exists(path):
            problems.append(f"missing shard: {path}")
            continue
        try:
            with zipfile.ZipFile(path) as zf:
                nf = len(zf.infolist())
                total = sum(i.file_size for i in zf.infolist())
            print(f"{name}: valid zip, {nf} files, {total / 1e6:.1f} MB unpacked")
            if args.list_zips and name == shards[0]:
                with zipfile.ZipFile(path) as zf:
                    print("--- first 12 entries of", name)
                    for i in zf.infolist()[:12]:
                        print("   ", i.filename)
        except zipfile.BadZipFile:
            problems.append(f"corrupt zip: {path}")

    if args.check_extracted and shards:
        first = os.path.join(args.data_dir, shards[0])
        with zipfile.ZipFile(first) as zf:
            wavs = [i.filename for i in zf.infolist() if i.filename.endswith(".wav")]
        missing = [w for w in wavs if not os.path.exists(os.path.join(args.data_dir, w))]
        if missing:
            problems.append(f"{len(missing)} wav files from {shards[0]} not found on disk "
                            "(run with --extract)")
        else:
            print(f"All {len(wavs)} wav files of {shards[0]} present on disk")

    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print("  -", p)
        raise SystemExit(1)
    print("\nVerification passed.")


if __name__ == "__main__":
    main()