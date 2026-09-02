#!/usr/bin/env python3
"""
build_index.py -- Build the SLR54 audio index for ASR.

Reads utt_spk_text.tsv, resolves FLAC paths from the extracted corpus,
and writes a TSV: audio_id, audio_location, reference.

Usage:
  python build_index.py
  python build_index.py --data-dir /path/to/extracted/slr54
  python build_index.py --limit 100   # quick test
"""

import argparse
import csv
import os
import sys


def load_index(tsv_path):
    entries = []
    with open(tsv_path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            file_id = parts[0].strip()
            speaker_id = parts[1].strip()
            reference = "\t".join(parts[2:]).strip()
            entries.append((file_id, speaker_id, reference))
    return entries


def resolve_flac(data_dir, file_id):
    """{data_dir}/asr_nepali_{shard}/asr_nepali/data/{prefix}/{file_id}.flac"""
    prefix = file_id[:2]
    shard = prefix[0]
    return os.path.join(data_dir, f"asr_nepali_{shard}",
                        "asr_nepali", "data", prefix, f"{file_id}.flac")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    person1 = os.path.dirname(script_dir)

    ap = argparse.ArgumentParser(description="Build SLR54 audio index for ASR.")
    ap.add_argument("--index", default=os.path.join(person1, "data", "slr54", "utt_spk_text.tsv"))
    ap.add_argument("--data-dir", default=os.path.join(person1, "..", "scripts", "data", "slr54"))
    ap.add_argument("--out", default=os.path.join(script_dir, "audio_index.tsv"))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    entries = load_index(args.index)
    print(f"Index: {len(entries)} entries from {args.index}", file=sys.stderr)

    rows = []
    found = 0
    missing = 0
    for file_id, speaker_id, reference in entries:
        flac = resolve_flac(data_dir, file_id)
        exists = os.path.exists(flac)
        if exists:
            found += 1
        else:
            missing += 1
        rows.append((file_id, flac, reference, exists))

    if args.limit:
        rows = rows[: args.limit]

    with open(args.out, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["audio_id", "audio_location", "reference", "flac_exists"])
        w.writerows(rows)

    n = len(rows)
    print(f"Wrote {args.out} ({n} rows, {found} found, {missing} missing)",
          file=sys.stderr)


if __name__ == "__main__":
    main()