#!/usr/bin/env python3
"""
lookup.py -- Standalone Nepali similar-word lookup.

Run from the nepali_similar_words/ folder:
  python lookup.py --word कल --k 5
  python lookup.py --word जिम्मेबारि --k 5
  python lookup.py --word कल --word सात --k 3
  python lookup.py --word कल --k 10 --threshold 0.03
  python lookup.py --word कल --json

Importable from scripts/:
  cd scripts && python -c "from similar_words import SimilarityIndex; ..."
"""

import argparse
import json
import os
import sys

os.environ.setdefault("PYTHONUTF8", "1")

# Resolve paths relative to THIS file
HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.join(HERE, "scripts")
DB = os.path.join(HERE, "db", "phoneme_db.pkl")

# Add scripts/ to path (same mechanism as original)
sys.path.insert(0, SCRIPTS)

from similar_words import SimilarityIndex


def format_result(results, idx, query):
    entry = idx.lookup(query)
    ipa_str = entry["ipa"] if entry else "?"
    lines = [f"Similar to '{query}' ({ipa_str}):"]
    for s in results:
        lines.append(
            f"  {s['word']:<12} d={s['dist']:.4f}  {s['ipa']:<24} "
            f"freq={s['freq']} utts={s['n_utts']} speakers={s['n_speakers']} "
            f"source={s['source']} file_ids={len(s['file_ids'])}"
        )
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Nepali similar-word lookup.")
    ap.add_argument("--db", default=DB)
    ap.add_argument("--word", action="append", default=None)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--radius", type=int, default=1)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if not args.word:
        ap.error("--word is required")

    idx = SimilarityIndex(args.db)
    print(f"Loaded {len(idx.entries)} words, {len(idx.buckets)} buckets",
          file=sys.stderr)

    if len(args.word) == 1:
        results = idx.get_similar(args.word[0], k=args.k,
                                  len_radius=args.radius,
                                  threshold=args.threshold)
        if args.json:
            print(json.dumps(results, ensure_ascii=False, indent=2))
        else:
            print(format_result(results, idx, args.word[0]))
    else:
        results = idx.get_similar_many(args.word, k=args.k,
                                       len_radius=args.radius,
                                       threshold=args.threshold)
        if args.json:
            print(json.dumps(results, ensure_ascii=False, indent=2))
        else:
            for w in args.word:
                if w in results:
                    print(format_result(results[w], idx, w))
                else:
                    print(f"No results for '{w}'")
                print()


if __name__ == "__main__":
    main()
