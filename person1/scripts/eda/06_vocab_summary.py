#!/usr/bin/env python3
"""EDA stage 06 — comprehensive vocabulary + text-level summary stats.

Vocabulary (reads artifacts/vocab.tsv):
  - vocab_summary.tsv       all words + freq + char-length + cleaned form + junk flag
  - length_stats.tsv        min/max/mean/median/percentiles of word length & freq
  - word_length_dist.tsv    lengthwise frequency (n_types, total freq, mean freq)
  - non_devanagari_tokens.tsv  tokens containing non-Devanagari characters
  - non_devanagari_chars.tsv   character-level frequency of non-Devanagari chars

Text / sentence level (reads data/slr54/utt_spk_text.tsv):
  - text_stats.tsv          utterances, chars, avg/median/min/max text length,
                            percentiles (chars and words), sentences per utterance
  - text_length_dist.tsv    histogram of text length in characters
  - sentence_stats.tsv      sentences per utterance, avg/max

Images:
  - images/vocab/length_percentiles.png
  - images/vocab/non_dev_char_bar.png
  - images/corpus/text_length_hist.png
  - images/corpus/sentences_per_utt_hist.png
"""

import collections
import math
import re
import statistics
import unicodedata

from common import ARTIFACTS, DATA_INDEX, read_tsv, savefig, setup_matplotlib, write_stats

plt = setup_matplotlib()

DEVANAGARI = re.compile(r"[\u0900-\u097F]")
DANDA = "।"  # U+0964
JUNK = re.compile(r"[^\u0900-\u097F\u0020]")  # non-Devanagari, non-space
JUNK_EDGE = re.compile(r"^[^\u0900-\u097F]+|[^\u0900-\u097F]+$")
ZWJ_ZWNJ = str.maketrans({"\u200D": "", "\u200C": ""})


def pct(values, p):
    s = sorted(values)
    if not s:
        return 0.0
    k = (len(s) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] + (s[c] - s[f]) * (k - f)


def clean_token(tok):
    tok = unicodedata.normalize("NFC", tok.strip()).translate(ZWJ_ZWNJ)
    return JUNK_EDGE.sub("", tok)


def main():
    rows = read_tsv(ARTIFACTS / "vocab.tsv")
    n_types = len(rows)
    total_tokens = sum(int(r["freq"]) for r in rows)

    # ---- vocabulary-level ----
    summary = []
    junk_tokens = []
    lengths = []
    freqs = []
    for i, r in enumerate(rows, 1):
        word = r["word"]
        cleaned = clean_token(word)
        jchars = sorted({c for c in word if not DEVANAGARI.match(c) and not c.isdigit()})
        has_junk = bool(jchars)
        lengths.append(len(word))
        freqs.append(int(r["freq"]))
        summary.append([i, word, r["freq"], r["n_utts"], r["n_speakers"],
                        len(word), cleaned, "yes" if has_junk else "no",
                        "".join(jchars)])
        if has_junk:
            junk_tokens.append([word, r["freq"], "".join(jchars), cleaned])
    write_stats("vocab", "vocab_summary.tsv",
                ["rank", "word", "freq", "n_utts", "n_speakers", "len_chars",
                 "cleaned_form", "has_non_devanagari", "non_devanagari_chars"],
                summary)
    write_stats("vocab", "non_devanagari_tokens.tsv",
                ["word", "freq", "non_devanagari_chars", "cleaned_form"], junk_tokens)

    len_stats = {
        "n_types": n_types,
        "len_min": min(lengths) if lengths else 0,
        "len_max": max(lengths) if lengths else 0,
        "len_mean": round(statistics.mean(lengths), 2) if lengths else 0,
        "len_median": int(pct(lengths, 0.5)),
        "len_p25": int(pct(lengths, 0.25)),
        "len_p75": int(pct(lengths, 0.75)),
        "len_p90": int(pct(lengths, 0.90)),
        "len_p99": int(pct(lengths, 0.99)),
        "freq_min": min(freqs) if freqs else 0,
        "freq_max": max(freqs) if freqs else 0,
        "freq_mean": round(statistics.mean(freqs), 2) if freqs else 0,
        "freq_median": int(pct(freqs, 0.5)),
        "freq_p90": int(pct(freqs, 0.90)),
        "freq_p99": int(pct(freqs, 0.99)),
        "total_tokens": total_tokens,
    }
    write_stats("vocab", "length_stats.tsv",
                ["metric", "value"], list(len_stats.items()))
    print("Vocabulary: {} types, {} tokens; length min={} max={} mean={}".format(
        n_types, total_tokens, len_stats["len_min"], len_stats["len_max"], len_stats["len_mean"]))

    # lengthwise frequency
    by_len = collections.defaultdict(lambda: [0, 0])  # len -> [n_types, tokens]
    for r, l in zip(summary, lengths):
        by_len[l][0] += 1
        by_len[l][1] += int(r[2])
    write_stats("vocab", "word_length_dist.tsv",
                ["len_chars", "n_types", "total_freq", "mean_freq"],
                [[l, v[0], v[1], round(v[1] / v[0], 2)] for l, v in sorted(by_len.items())])

    # length distribution figure with percentile markers
    counts = [v[0] for _, v in sorted(by_len.items())]
    xs = list(range(1, len(counts) + 1))
    plt.figure(figsize=(9, 4))
    plt.bar(xs, counts, color="#4C72B0")
    for name, p, color in [("P50", 0.50, "#C44E52"), ("P90", 0.90, "#DD8452"),
                           ("P99", 0.99, "#8172B3")]:
        v = int(pct(lengths, p))
        plt.axvline(v, color=color, ls="--", lw=1.2)
        plt.text(v, max(counts) * 0.95, f"{name}={v}", color=color, rotation=90)
    plt.xlabel("word length (characters)")
    plt.ylabel("types")
    plt.title("Word length distribution with percentiles")
    print("Wrote", savefig(plt, "vocab", "length_percentiles.png"))

    # non-Devanagari character breakdown
    char_ct = collections.Counter()
    for r, has in zip(summary, lengths):
        word = r[1]
        if r[7] == "yes":
            for c in word:
                if not DEVANAGARI.match(c) and not c.isdigit():
                    char_ct[c] += 1
    n_chars_total = sum(char_ct.values())
    write_stats("chars", "non_devanagari_chars.tsv",
                ["char", "codepoint", "name", "count", "pct_of_nondev"],
                [[c, "U+{:04X}".format(ord(c)), unicodedata.name(c, "?"), n,
                  round(100 * n / n_chars_total, 2)] if n_chars_total else
                 [c, "U+{:04X}".format(ord(c)), unicodedata.name(c, "?"), n, 0.0]
                 for c, n in char_ct.most_common()])
    if char_ct:
        top = char_ct.most_common(25)
        plt.figure(figsize=(10, 4))
        plt.bar(range(len(top)), [n for _, n in top], color="#C44E52")
        plt.xticks(range(len(top)), [c for c, _ in top])
        plt.xlabel("non-Devanagari character")
        plt.ylabel("occurrences")
        plt.title(f"Non-Devanagari characters in vocab ({n_chars_total} total)")
        print("Wrote", savefig(plt, "vocab", "non_dev_char_bar.png"))
    else:
        print("No non-Devanagari characters found in vocab.")

    # ---- text / sentence level ----
    text_lens_chars = []     # chars incl. spaces
    text_lens_words = []     # word counts
    sentences = []           # sentences per utterance
    n_utts = 0
    n_chars = 0
    n_sentences = 0
    with open(DATA_INDEX, encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            transcript = "\t".join(parts[2:]).strip()
            n_utts += 1
            n_chars += len(transcript)
            text_lens_chars.append(len(transcript))
            text_lens_words.append(len(transcript.split()))
            ns = transcript.count(DANDA) + (1 if transcript else 0)
            sentences.append(ns)
            n_sentences += ns

    text_stats = {
        "utterances": n_utts,
        "total_chars": n_chars,
        "avg_text_len_chars": round(statistics.mean(text_lens_chars), 2),
        "median_text_len_chars": int(pct(text_lens_chars, 0.5)),
        "min_text_len_chars": min(text_lens_chars),
        "max_text_len_chars": max(text_lens_chars),
        "p25_text_len_chars": int(pct(text_lens_chars, 0.25)),
        "p75_text_len_chars": int(pct(text_lens_chars, 0.75)),
        "p90_text_len_chars": int(pct(text_lens_chars, 0.90)),
        "p99_text_len_chars": int(pct(text_lens_chars, 0.99)),
        "avg_text_len_words": round(statistics.mean(text_lens_words), 2),
        "median_text_len_words": int(pct(text_lens_words, 0.5)),
        "min_text_len_words": min(text_lens_words),
        "max_text_len_words": max(text_lens_words),
        "total_sentences": n_sentences,
        "avg_sentences_per_utt": round(statistics.mean(sentences), 3),
        "max_sentences_per_utt": max(sentences),
    }
    write_stats("corpus", "text_stats.tsv", ["metric", "value"],
                list(text_stats.items()))
    print("Text: {} utterances, avg {:.1f} chars, avg {:.1f} words".format(
        n_utts, text_stats["avg_text_len_chars"], text_stats["avg_text_len_words"]))

    # text length histogram (chars)
    bins = list(range(0, max(text_lens_chars) + 51, 50))
    plt.figure()
    plt.hist(text_lens_chars, bins=bins, color="#55A868")
    for p, color in [(0.5, "#C44E52"), (0.9, "#DD8452"), (0.99, "#8172B3")]:
        v = int(pct(text_lens_chars, p))
        plt.axvline(v, color=color, ls="--", lw=1.2)
        plt.text(v, plt.ylim()[1] * 0.95, f"P{int(p*100)}={v}", color=color, rotation=90)
    plt.xlabel("text length (characters)")
    plt.ylabel("utterances")
    plt.title("Text length distribution (chars) with percentiles")
    print("Wrote", savefig(plt, "corpus", "text_length_hist.png"))

    # sentences per utterance histogram
    sent_dist = collections.Counter(sentences)
    xs_s = sorted(sent_dist)
    write_stats("corpus", "sentence_stats.tsv", ["sentences_per_utt", "n_utts"],
                [[x, sent_dist[x]] for x in xs_s])
    plt.figure()
    plt.bar([str(x) for x in xs_s[:20]], [sent_dist[x] for x in xs_s[:20]], color="#55A868")
    plt.xlabel("sentences per utterance")
    plt.ylabel("utterances")
    plt.title("Sentences per utterance")
    plt.xticks(rotation=45)
    print("Wrote", savefig(plt, "corpus", "sentences_per_utt_hist.png"))

    print("\nVocab + text summary complete.")


if __name__ == "__main__":
    main()