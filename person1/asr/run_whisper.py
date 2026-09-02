#!/usr/bin/env python3
"""
run_whisper.py -- Run Whisper ASR on all SLR54 audio files.

Reads audio_index.tsv (from build_index.py), transcribes each FLAC file
via faster-whisper (large-v3-turbo, GPU, batched), and writes results
to two TSVs (utterance-level + word-level).

Usage:
  python run_whisper.py                    # full corpus
  python run_whisper.py --limit 50         # test on first 50
  python run_whisper.py --resume           # skip already-done files
"""

import argparse
import csv
import os
import sys
import time

from whisper.transcribe import WhisperASR

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def load_index(index_path):
    rows = []
    with open(index_path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            rows.append(row)
    return rows


def load_done(out_path):
    done = set()
    if not os.path.exists(out_path):
        return done
    with open(out_path, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            done.add(row["audio_id"])
    return done


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    ap = argparse.ArgumentParser(description="Run Whisper ASR on SLR54 corpus.")
    ap.add_argument("--index", default=os.path.join(script_dir, "audio_index.tsv"))
    ap.add_argument("--out-dir", default=script_dir)
    ap.add_argument("--model", default="large-v3-turbo")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--language", default="ne")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    out_utt = os.path.join(args.out_dir, "whisper_output.tsv")
    out_words = os.path.join(args.out_dir, "whisper_words.tsv")

    rows = load_index(args.index)
    print(f"Index: {len(rows)} utterances", file=sys.stderr)

    existing = [r for r in rows if r.get("flac_exists") == "True"
                or os.path.exists(r["audio_location"])]
    print(f"Found {len(existing)} FLAC files", file=sys.stderr)

    if args.limit:
        existing = existing[: args.limit]
        print(f"Limit: processing first {args.limit} files", file=sys.stderr)

    done_ids = load_done(out_utt)
    if done_ids:
        existing = [r for r in existing if r["audio_id"] not in done_ids]
        print(f"Resume: {len(done_ids)} already done, {len(existing)} remaining",
              file=sys.stderr)

    if not existing:
        print("Nothing to do.", file=sys.stderr)
        return

    asr = WhisperASR(model_size=args.model, batch_size=args.batch_size)

    utt_header = ["audio_id", "reference", "hypothesis",
                  "avg_logprob", "no_speech_prob", "duration",
                  "inference_time", "n_segments"]
    words_header = ["audio_id", "word_index", "word", "start", "end",
                    "probability", "segment_index", "segment_text"]

    write_utt_header = not os.path.exists(out_utt) or os.path.getsize(out_utt) == 0
    write_words_header = not os.path.exists(out_words) or os.path.getsize(out_words) == 0

    fh_utt = open(out_utt, "a", encoding="utf-8", newline="")
    fh_words = open(out_words, "a", encoding="utf-8", newline="")
    w_utt = csv.writer(fh_utt, delimiter="\t")
    w_words = csv.writer(fh_words, delimiter="\t")
    if write_utt_header:
        w_utt.writerow(utt_header)
    if write_words_header:
        w_words.writerow(words_header)

    total_t0 = time.time()
    n_done = 0
    n_error = 0

    iterator = existing
    if tqdm is not None:
        iterator = tqdm(existing, unit="utt", desc="Whisper ASR",
                        dynamic_ncols=True)

    for row in iterator:
        audio_id = row["audio_id"]
        audio_path = row["audio_location"]
        reference = row["reference"]

        try:
            result = asr.transcribe_file(audio_path, language=args.language)
            words = result["words"]
            segments = result["segments"]

            w_utt.writerow([
                audio_id, reference, result["hypothesis"],
                result["avg_logprob"], result["no_speech_prob"],
                result["duration"], result["inference_time"],
                len(segments),
            ])
            fh_utt.flush()

            for wi, w in enumerate(words):
                w_words.writerow([
                    audio_id, wi, w["word"], w["start"], w["end"],
                    w["probability"], w["segment_idx"],
                    segments[w["segment_idx"]]["text"],
                ])
            fh_words.flush()

            n_done += 1
        except Exception as exc:
            w_utt.writerow([audio_id, reference, f"ERROR: {exc}",
                            "", "", "", "", ""])
            fh_utt.flush()
            n_error += 1

    fh_utt.close()
    fh_words.close()
    total_time = time.time() - total_t0
    print(f"\nDone: {n_done} transcribed, {n_error} errors, "
          f"{total_time:.0f}s total ({total_time / max(n_done, 1):.2f}s/utt)",
          file=sys.stderr)
    print(f"Utterances: {out_utt}", file=sys.stderr)
    print(f"Words:      {out_words}", file=sys.stderr)


if __name__ == "__main__":
    main()
