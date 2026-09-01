"""
nepali_similar_words -- Find similar-sounding Nepali words from a phoneme DB.

Usage:
    from nepali_similar_words import SimilarityIndex

    idx = SimilarityIndex()                    # auto-finds db/phoneme_db.pkl
    idx.get_similar("कल", k=5)
    idx.get_similar("जिम्मेबारि", k=5)
    idx.get_similar_many(["कल", "सात"], k=5)
    idx.lookup("कल")
"""

import os
import sys
import importlib.util

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS_DIR = os.path.join(_PKG_DIR, "scripts")
_DB_PATH = os.path.join(_PKG_DIR, "db", "phoneme_db.pkl")


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_ipa_utils = _load_module("ipa_utils", os.path.join(_SCRIPTS_DIR, "ipa_utils.py"))
_word_stats = _load_module("word_stats", os.path.join(_SCRIPTS_DIR, "word_stats.py"))
_phonemize = _load_module("phonemize", os.path.join(_SCRIPTS_DIR, "phonemize.py"))
_phonetic_distance = _load_module("phonetic_distance",
                                   os.path.join(_SCRIPTS_DIR, "phonetic_distance.py"))
_similar_words = _load_module("similar_words",
                               os.path.join(_SCRIPTS_DIR, "similar_words.py"))

_BaseIndex = _similar_words.SimilarityIndex


class SimilarityIndex:
    def __init__(self, db_path=None):
        if db_path is None:
            db_path = _DB_PATH
        self._inner = _BaseIndex(db_path)

    def get_similar(self, word, k=5, len_radius=1, threshold=None):
        return self._inner.get_similar(word, k=k, len_radius=len_radius,
                                       threshold=threshold)

    def get_similar_many(self, words, k=5, len_radius=1, threshold=None,
                         jobs=None):
        return self._inner.get_similar_many(words, k=k, len_radius=len_radius,
                                            threshold=threshold, jobs=jobs)

    def lookup(self, word):
        return self._inner.lookup(word)

    @property
    def word_count(self):
        return len(self._inner.entries)

    @property
    def bucket_count(self):
        return len(self._inner.buckets)

    def __repr__(self):
        return (f"SimilarityIndex(words={self.word_count}, "
                f"buckets={self.bucket_count})")
