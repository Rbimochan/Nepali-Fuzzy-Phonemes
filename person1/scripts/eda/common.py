#!/usr/bin/env python3
"""Shared paths, fonts, and helpers for Person-1 EDA scripts."""

import csv
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # person1/
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))  # make ipa_utils importable from eda/ scripts

DATA_INDEX = ROOT / "data" / "slr54" / "utt_spk_text.tsv"
ARTIFACTS = ROOT / "artifacts"
STATS = ROOT / "stats"
IMAGES = ROOT / "images"

DEVANAGARI_FONT_CANDIDATES = [
    "Nirmala UI", "Mangal", "Kokila", "Arial", "Segoe UI",
    "Noto Sans Devanagari", "DejaVu Sans",
]


def setup_matplotlib():
    """Configure matplotlib with a Devanagari-capable font if available."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    available = {f.name for f in font_manager.fontManager.ttflist}
    font = next((c for c in DEVANAGARI_FONT_CANDIDATES if c in available), None)
    if font:
        plt.rcParams["font.family"] = font
    plt.rcParams["axes.titlesize"] = 11
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.bbox"] = "tight"
    return plt


def savefig(plt, subdir, name):
    """Save a figure into images/<subdir>/<name>."""
    out = IMAGES / subdir
    os.makedirs(out, exist_ok=True)
    path = out / name
    plt.savefig(path)
    plt.close()
    return path


def write_stats(subdir, name, header, rows):
    """Write a stats table into stats/<subdir>/<name>."""
    out = STATS / subdir
    os.makedirs(out, exist_ok=True)
    path = out / name
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(header)
        w.writerows(rows)
    return path


def read_tsv(path):
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))