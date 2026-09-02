#!/usr/bin/env python3
"""Run the full Person-1 pipeline + EDA. Execute from person1/.

Order:
  1. word_stats.py      -> artifacts/vocab.tsv, artifacts/corpus_stats.tsv
  2. phonemize.py       -> artifacts/phonemes.tsv (full vocab G2P)
  3. phoneme_db.py      -> db/phoneme_db.{tsv,pkl},
                           db/phoneme_occurrences.tsv
  4. similar_words.py --dump -> artifacts/similar_words.tsv
  5. EDA stages 01-06   -> stats/<component>/ + images/<component>/

Usage:
  python scripts/eda/run_all.py [--min-freq N] [--max-vocab N] [--skip-pipeline]
"""

import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPTS = os.path.join(ROOT, "scripts")
PY = sys.executable


def run(*args):
    print("\n=== " + " ".join(args) + " ===")
    subprocess.run([PY] + list(args), check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-freq", type=int, default=1)
    ap.add_argument("--max-vocab", type=int, default=None,
                    help="limit phonemization + DB build to first N vocab words")
    ap.add_argument("--skip-pipeline", action="store_true",
                    help="reuse existing artifacts, run EDA only")
    ap.add_argument("--dump-jobs", type=int, default=None,
                    help="parallel workers for similar_words --dump")
    args = ap.parse_args()

    if not args.skip_pipeline:
        run(os.path.join(SCRIPTS, "word_stats.py"),
            "--index", os.path.join(ROOT, "data", "slr54", "utt_spk_text.tsv"),
            "--out-dir", os.path.join(ROOT, "artifacts"))
        run(os.path.join(SCRIPTS, "phonemize.py"),
            "--words", os.path.join(ROOT, "artifacts", "vocab.tsv"),
            "--out", os.path.join(ROOT, "artifacts", "phonemes.tsv"),
            *(("--limit", str(args.max_vocab)) if args.max_vocab else ()))
        run(os.path.join(SCRIPTS, "phoneme_db.py"),
            "--vocab", os.path.join(ROOT, "artifacts", "vocab.tsv"),
            "--phonemes", os.path.join(ROOT, "artifacts", "phonemes.tsv"),
            "--index", os.path.join(ROOT, "data", "slr54", "utt_spk_text.tsv"),
            "--db-dir", os.path.join(ROOT, "db"))
        run(os.path.join(SCRIPTS, "similar_words.py"), "--dump",
            "--db", os.path.join(ROOT, "db", "phoneme_db.pkl"),
            "--out", os.path.join(ROOT, "artifacts", "similar_words.tsv"),
            *(("--jobs", str(args.dump_jobs)) if args.dump_jobs else ()))

    eda = os.path.join(SCRIPTS, "eda")
    for stage in ["01_corpus_eda.py", "02_vocab_eda.py", "03_char_eda.py",
                  "04_phoneme_eda.py", "05_similar_words_eda.py",
                  "06_vocab_summary.py"]:
        run(os.path.join(eda, stage))
    print("\nAll EDA done. Stats in person1/stats/, images in person1/images/.")


if __name__ == "__main__":
    main()