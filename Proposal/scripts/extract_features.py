"""QC-normalize pilot-pair clips and extract acoustic features.

Reads clips from Proposal/clips/<pair_id>/*.wav (filename convention:
<speaker>_<word>_<label>.wav, label = the phoneme side of the pair, e.g.
"ba" or "bha"), resamples to 16kHz mono, trims leading/trailing silence,
and extracts MFCCs + deltas, spectral centroid (sibilant-pair proxy),
duration, and RMS-onset-based VOT proxy. Writes one feature manifest CSV
across all pairs found under Proposal/clips/.

Usage:
    python Proposal/scripts/extract_features.py \
        --clips-dir Proposal/clips \
        --out Proposal/artifacts/feature_manifest.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import librosa
import numpy as np

TARGET_SR = 16000
N_MFCC = 13


def qc_normalize(path: Path) -> tuple[np.ndarray, int]:
    y, sr = librosa.load(str(path), sr=TARGET_SR, mono=True)
    y_trimmed, _ = librosa.effects.trim(y, top_db=30)
    # top_db=30 can over-trim short/quiet-tailed words and clip real
    # speech (seen on fast words like "taar" and "mool" — the raw clip had
    # a clear voice signal, but trim cut it to a fraction of a second). If
    # trimming would keep less than half the original audio, the original
    # signal likely wasn't the quiet room-noise this is meant to strip —
    # keep the untrimmed clip instead of losing the word.
    if len(y_trimmed) < 0.5 * len(y):
        return y, sr
    return y_trimmed, sr


def extract_features(y: np.ndarray, sr: int) -> dict:
    duration = len(y) / sr

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    # delta's default width=9 needs >=9 frames; very short clips have
    # fewer, so shrink to the largest odd width that fits (min 3).
    n_frames = mfcc.shape[1]
    delta_width = min(9, n_frames if n_frames % 2 == 1 else n_frames - 1)
    delta_width = max(3, delta_width)
    if n_frames >= delta_width:
        mfcc_delta = librosa.feature.delta(mfcc, width=delta_width)
        mfcc_delta_mean = mfcc_delta.mean(axis=1)
    else:
        # fewer than 3 frames total — can't compute a delta at all
        mfcc_delta_mean = np.zeros(N_MFCC)
    mfcc_mean = mfcc.mean(axis=1)

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    centroid_mean = float(centroid.mean()) if centroid.size else 0.0

    # VOT proxy: time from onset to first RMS-energy peak past a threshold,
    # not a substitute for forced-aligned VOT but usable for the pilot.
    rms = librosa.feature.rms(y=y)[0]
    if rms.size and rms.max() > 0:
        onset_frame = int(np.argmax(rms > 0.3 * rms.max()))
        vot_proxy = onset_frame * (512 / sr)  # hop_length default 512
    else:
        vot_proxy = 0.0

    feats = {
        "duration_s": duration,
        "spectral_centroid_hz": centroid_mean,
        "vot_proxy_s": vot_proxy,
    }
    for i, v in enumerate(mfcc_mean):
        feats[f"mfcc{i}_mean"] = float(v)
    for i, v in enumerate(mfcc_delta_mean):
        feats[f"mfcc{i}_delta_mean"] = float(v)
    return feats


def parse_filename(path: Path) -> tuple[str, str, str]:
    """<speaker>_<word>_<label>.wav -> (speaker, word, label)."""
    stem = path.stem
    parts = stem.split("_")
    speaker = parts[0]
    label = parts[-1]
    word = "_".join(parts[1:-1])
    return speaker, word, label


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips-dir", default="Proposal/clips")
    ap.add_argument("--out", default="Proposal/artifacts/feature_manifest.csv")
    args = ap.parse_args()

    clips_dir = Path(args.clips_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    feature_keys: list[str] | None = None

    for pair_dir in sorted(clips_dir.iterdir()):
        if not pair_dir.is_dir():
            continue
        pair_id = pair_dir.name
        wavs = sorted(pair_dir.glob("*.wav"))
        if not wavs:
            continue
        for wav_path in wavs:
            speaker, word, label = parse_filename(wav_path)
            y, sr = qc_normalize(wav_path)
            feats = extract_features(y, sr)
            if feature_keys is None:
                feature_keys = list(feats.keys())
            row = {
                "pair_id": pair_id,
                "speaker": speaker,
                "word": word,
                "label": label,
                "file": str(wav_path),
                **feats,
            }
            rows.append(row)
            print(f"[ok] {pair_id}/{wav_path.name} -> label={label}")

    if not rows:
        print("No clips found under", clips_dir)
        return

    fieldnames = ["pair_id", "speaker", "word", "label", "file"] + (feature_keys or [])
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
