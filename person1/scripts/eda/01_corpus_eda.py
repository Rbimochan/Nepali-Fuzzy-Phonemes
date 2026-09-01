#!/usr/bin/env python3
"""EDA stage 01 — corpus-level stats from the SLR54 index.

Stats  : stats/corpus/corpus_stats.tsv, utterance length histogram,
         speakers-per-utterance histogram
Images : images/corpus/utt_length_hist.png, speaker_utt_hist.png,
         speaker_pareto.png
"""

import collections
import os

from common import DATA_INDEX, savefig, setup_matplotlib, write_stats

plt = setup_matplotlib()


def main():
    utt_lengths = []          # words per utterance
    speaker_utts = collections.Counter()
    speakers = set()
    tokens = 0
    vocab = set()

    with open(DATA_INDEX, encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            file_id, user_id = parts[0], parts[1]
            transcript = "\t".join(parts[2:]).split()
            n = len(transcript)
            utt_lengths.append(n)
            tokens += n
            vocab.update(transcript)
            speakers.add(user_id)
            speaker_utts[user_id] += 1

    n_utts = len(utt_lengths)
    avg = sum(utt_lengths) / n_utts if n_utts else 0.0

    rows = [("utterances", n_utts), ("speakers", len(speakers)),
            ("tokens", tokens), ("vocab_types", len(vocab)),
            ("tokens_per_utterance_mean", round(avg, 2)),
            ("tokens_per_utterance_median", sorted(utt_lengths)[n_utts // 2])]
    path = write_stats("corpus", "corpus_stats.tsv", ["metric", "value"], rows)
    print("Wrote", path)

    # Utterance length histogram
    max_len = max(utt_lengths)
    bins = range(0, max_len + 2, 2)
    counts, edges = _hist(utt_lengths, bins)
    write_stats("corpus", "utt_length_hist.tsv",
                ["len_bin_low", "len_bin_high", "utterances"],
                [[lo, hi, c] for (lo, hi), c in zip(zip(edges[:-1], edges[1:]), counts)])
    plt.figure()
    plt.hist(utt_lengths, bins=bins, color="#4C72B0")
    plt.xlabel("words per utterance")
    plt.ylabel("utterances")
    plt.title("Utterance length distribution")
    print("Wrote", savefig(plt, "corpus", "utt_length_hist.png"))

    # Speakers per utterance-count histogram
    dist = collections.Counter(speaker_utts.values())
    xs = sorted(dist)
    write_stats("corpus", "speaker_utt_dist.tsv", ["utterances_per_speaker", "n_speakers"],
                [[x, dist[x]] for x in xs])
    plt.figure()
    plt.bar([str(x) for x in xs[:40]], [dist[x] for x in xs[:40]], color="#55A868")
    plt.xlabel("utterances per speaker")
    plt.ylabel("speakers")
    plt.title("Utterances per speaker (top 40 bins)")
    plt.xticks(rotation=45)
    print("Wrote", savefig(plt, "corpus", "speaker_utt_hist.png"))

    # Pareto: cumulative share of utterances vs speakers
    cum = 0
    total = n_utts
    pareto = []
    for rank, (uid, cnt) in enumerate(sorted(speaker_utts.items(), key=lambda x: -x[1]), 1):
        cum += cnt
        pareto.append([rank, cnt, cum / total])
    write_stats("corpus", "speaker_pareto.tsv",
                ["rank", "utterances", "cum_share"], pareto)
    plt.figure()
    plt.plot([p[0] for p in pareto], [p[2] * 100 for p in pareto], color="#C44E52")
    plt.xlabel("speaker rank (most active first)")
    plt.ylabel("cumulative % of utterances")
    plt.title("Speaker activity (Pareto)")
    plt.grid(alpha=0.3)
    print("Wrote", savefig(plt, "corpus", "speaker_pareto.png"))


def _hist(values, bins):
    counts = [0] * (len(bins) - 1)
    for v in values:
        for i in range(len(bins) - 1):
            if bins[i] <= v < bins[i + 1]:
                counts[i] += 1
                break
        else:
            counts[-1] += 1
    return counts, bins


if __name__ == "__main__":
    main()