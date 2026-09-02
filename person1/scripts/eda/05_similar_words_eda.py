#!/usr/bin/env python3
"""EDA stage 05 — similar-word dump stats
(requires artifacts/similar_words.tsv from similar_words.py --dump).

Columns: word, rank, similar_word, dist, ipa, freq, n_utts

Stats  : stats/similar_words/distance_dist.tsv, rank_dist.tsv,
         coverage.tsv, summary.tsv, example_similar.tsv
Images : images/similar_words/distance_hist.png, rank_dist.png,
         coverage_bar.png
"""

import collections

from common import ARTIFACTS, read_tsv, savefig, setup_matplotlib, write_stats

plt = setup_matplotlib()

MAX_RANK = 5  # dump default k


def main():
    rows = read_tsv(ARTIFACTS / "similar_words.tsv")
    if not rows:
        print("similar_words.tsv is empty - run similar_words.py --dump first")
        return

    dists = [float(r["dist"]) for r in rows]
    write_stats("similar_words", "distance_dist.tsv", ["dist_bin", "n_rows"],
                _binned(dists, 0.0, 0.25, 0.005))
    plt.figure()
    plt.hist(dists, bins=50, color="#4C72B0")
    plt.xlabel("normalized PanPhon distance")
    plt.ylabel("similar-word rows")
    plt.title(f"Similar-word distance distribution (n={len(dists)})")
    print("Wrote", savefig(plt, "similar_words", "distance_hist.png"))

    # rank statistics: how distance grows with rank
    by_rank = collections.defaultdict(list)
    for r in rows:
        by_rank[int(r["rank"])].append(float(r["dist"]))
    rank_rows = []
    for rank in sorted(by_rank):
        vs = by_rank[rank]
        rank_rows.append([rank, len(vs), round(sum(vs) / len(vs), 4),
                          round(sorted(vs)[len(vs) // 2], 4),
                          round(min(vs), 4), round(max(vs), 4)])
    write_stats("similar_words", "rank_dist.tsv",
                ["rank", "n_rows", "mean_dist", "median_dist", "min_dist", "max_dist"],
                rank_rows)
    plt.figure()
    ranks = [r[0] for r in rank_rows]
    means = [r[2] for r in rank_rows]
    meds = [r[3] for r in rank_rows]
    plt.plot(ranks, means, marker="o", label="mean")
    plt.plot(ranks, meds, marker="s", label="median")
    plt.xlabel("rank")
    plt.ylabel("normalized distance")
    plt.title("Distance by rank (similar-word results)")
    plt.legend()
    print("Wrote", savefig(plt, "similar_words", "rank_dist.png"))

    # coverage: how many query words got any similar word
    queried = {r["word"] for r in rows}
    with_hits = {r["word"] for r in rows if int(r["rank"]) == 1}
    no_hits = queried - with_hits
    write_stats("similar_words", "coverage.tsv", ["metric", "value"],
                [["query_words", len(queried)],
                 ["with_similar_words", len(with_hits)],
                 ["without_similar_words", len(no_hits)],
                 ["total_rows", len(rows)]])
    plt.figure()
    plt.bar(["with similar", "without"], [len(with_hits), len(no_hits)],
            color=["#55A868", "#C44E52"])
    plt.ylabel("query words")
    plt.title("Coverage of similar-word lookup")
    print("Wrote", savefig(plt, "similar_words", "coverage_bar.png"))

    # examples: pick some well-known words with several similar results
    examples = ["कल", "सात", "यदि", "काल", "रात", "पानी"]
    ex_rows = []
    for w in examples:
        hits = [r for r in rows if r["word"] == w and int(r["rank"]) <= MAX_RANK]
        for r in hits:
            ex_rows.append([w, r["rank"], r["similar_word"], r["dist"],
                            r["ipa"], r["freq"], r["n_utts"]])
    write_stats("similar_words", "example_similar.tsv",
                ["word", "rank", "similar_word", "dist", "ipa", "freq", "n_utts"],
                ex_rows)
    print(f"Similar words: {len(rows)} rows for {len(queried)} query words")


def _binned(values, lo, hi, step):
    out = collections.OrderedDict()
    x = lo
    while x < hi:
        out[(round(x, 4), round(x + step, 4))] = 0
        x += step
    for v in values:
        for (a, b) in out:
            if a <= v < b:
                out[(a, b)] += 1
                break
    return [[a, b, c] for (a, b), c in out.items()]


if __name__ == "__main__":
    main()