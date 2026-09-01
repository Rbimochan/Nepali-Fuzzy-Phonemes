#!/usr/bin/env python3
"""EDA stage 02 — vocabulary-level stats (requires artifacts/vocab.tsv).

Stats  : stats/vocab/word_length_dist.tsv, freq_bands.tsv, top_words.tsv
Images : images/vocab/zipf.png, word_length_hist.png, freq_band_bar.png
"""

import collections
import math

from common import ARTIFACTS, read_tsv, savefig, setup_matplotlib, write_stats

plt = setup_matplotlib()


def main():
    rows = read_tsv(ARTIFACTS / "vocab.tsv")
    n_types = len(rows)
    total_tokens = sum(int(r["freq"]) for r in rows)

    # Rank-frequency (Zipf)
    zipf = [[i + 1, int(r["freq"])] for i, r in enumerate(rows)]
    write_stats("vocab", "zipf_rank_freq.tsv", ["rank", "freq"], zipf)
    plt.figure()
    plt.loglog([z[0] for z in zipf], [z[1] for z in zipf], ".", markersize=3, color="#4C72B0")
    plt.xlabel("frequency rank")
    plt.ylabel("frequency")
    plt.title(f"Zipf plot ({n_types:,} types)")
    plt.grid(alpha=0.3, which="both")
    print("Wrote", savefig(plt, "vocab", "zipf.png"))

    # Word length distribution (in Devanagari characters)
    lens = collections.Counter(len(r["word"]) for r in rows)
    write_stats("vocab", "word_length_dist.tsv", ["word_len_chars", "n_types"],
                [[l, lens[l]] for l in sorted(lens)])
    plt.figure()
    plt.bar(list(lens.keys()), list(lens.values()), color="#55A868")
    plt.xlabel("word length (characters)")
    plt.ylabel("types")
    plt.title("Word length distribution")
    print("Wrote", savefig(plt, "vocab", "word_length_hist.png"))

    # Frequency bands
    bands = [("1 (hapax)", 1, 1), ("2", 2, 2), ("3-5", 3, 5), ("6-10", 6, 10),
             ("11-50", 11, 50), ("51-200", 51, 200), ("201+", 201, None)]
    band_rows = []
    for label, lo, hi in bands:
        cnt = sum(1 for r in rows if int(r["freq"]) >= lo and (hi is None or int(r["freq"]) <= hi))
        band_rows.append([label, cnt, round(100 * cnt / n_types, 2)])
    write_stats("vocab", "freq_bands.tsv", ["band", "n_types", "pct_types"], band_rows)
    plt.figure()
    plt.bar([b[0] for b in band_rows], [b[1] for b in band_rows], color="#C44E52")
    plt.yscale("log")
    plt.xlabel("frequency band")
    plt.ylabel("types (log)")
    plt.title("Vocabulary frequency bands")
    print("Wrote", savefig(plt, "vocab", "freq_band_bar.png"))

    # Top words + coverage share
    top = rows[:50]
    write_stats("vocab", "top_words.tsv", ["rank", "word", "freq", "n_utts", "n_speakers", "cum_share"],
                [[i, r["word"], r["freq"], r["n_utts"], r["n_speakers"],
                  round(sum(int(t["freq"]) for t in rows[:i]) / total_tokens, 4)]
                 for i, r in enumerate(top, 1)])
    print(f"Vocab: {n_types:,} types, {total_tokens:,} tokens")


if __name__ == "__main__":
    main()