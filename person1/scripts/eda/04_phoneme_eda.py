#!/usr/bin/env python3
"""EDA stage 04 — phoneme-level stats from the G2P output
(requires artifacts/phonemes.tsv from phonemize.py).

Stats  : stats/phonemes/phoneme_freq.tsv, ipa_word_length.tsv,
         g2p_coverage.tsv
Images : images/phonemes/phoneme_freq_top.png, ipa_len_hist.png
"""

import collections

import panphon

from common import ARTIFACTS, read_tsv, savefig, setup_matplotlib, write_stats
from ipa_utils import normalize_affricates

plt = setup_matplotlib()
ft = panphon.FeatureTable()


def main():
    rows = read_tsv(ARTIFACTS / "phonemes.tsv")
    ok = [r for r in rows if r["status"] == "ok"]
    n_total, n_ok = len(rows), len(ok)

    write_stats("phonemes", "g2p_coverage.tsv", ["metric", "value"],
                [["words_total", n_total], ["words_ok", n_ok],
                 ["coverage", round(n_ok / n_total, 4)]])

    phones = collections.Counter()
    lengths = []
    for r in ok:
        segs = ft.ipa_segs(normalize_affricates(r["ipa"]))
        phones.update(segs)
        lengths.append(len(segs))

    total = sum(phones.values())
    table = sorted(phones.items(), key=lambda x: -x[1])
    write_stats("phonemes", "phoneme_freq.tsv",
                ["phoneme", "count", "pct"],
                [[p, c, round(100 * c / total, 3)] for p, c in table])
    print(f"IPA phoneme inventory: {len(table)} segments")

    top = table[:40]
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(top)), [c for _, c in top], color="#55A868")
    plt.xticks(range(len(top)), [p for p, _ in top])
    plt.xlabel("phoneme (IPA)")
    plt.ylabel("count")
    plt.title(f"Top 40 phonemes from G2P ({total:,} segments)")
    print("Wrote", savefig(plt, "phonemes", "phoneme_freq_top.png"))

    lens = collections.Counter(lengths)
    write_stats("phonemes", "ipa_word_length.tsv", ["n_phonemes", "n_words"],
                [[l, lens[l]] for l in sorted(lens)])
    plt.figure()
    plt.bar(list(lens.keys()), list(lens.values()), color="#C44E52")
    plt.xlabel("word length (phonemes)")
    plt.ylabel("words")
    plt.title("IPA word length distribution")
    print("Wrote", savefig(plt, "phonemes", "ipa_len_hist.png"))


if __name__ == "__main__":
    main()