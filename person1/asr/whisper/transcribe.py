#!/usr/bin/env python3
"""
transcribe.py -- Core Whisper transcription API for Nepali (SLR54).

Uses faster-whisper with large-v3-turbo on GPU via BatchedInferencePipeline.

Usage (importable):
    from transcribe import WhisperASR
    asr = WhisperASR()
    result = asr.transcribe_file(path)
"""

import sys
import time
import threading

from faster_whisper import BatchedInferencePipeline, WhisperModel


class WhisperASR:
    """Whisper ASR engine (large-v3-turbo, GPU, batched)."""

    def __init__(self, model_size="large-v3-turbo", device="cuda",
                 compute_type="float16", batch_size=16):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.batch_size = batch_size
        self._pipeline = None
        self._lock = threading.Lock()

    def _load(self):
        if self._pipeline is None:
            with self._lock:
                if self._pipeline is None:
                    t0 = time.time()
                    model = WhisperModel(self.model_size, device=self.device,
                                         compute_type=self.compute_type)
                    self._pipeline = BatchedInferencePipeline(model)
                    print(f"Whisper model '{self.model_size}' loaded in "
                          f"{time.time() - t0:.1f}s on {self.device}",
                          file=sys.stderr)

    def transcribe_file(self, audio_path, language="ne"):
        """Transcribe a single audio file with word-level timestamps."""
        self._load()
        t0 = time.time()
        segments_iter, info = self._pipeline.transcribe(
            audio_path,
            language=language,
            vad_filter=False,
            batch_size=self.batch_size,
            beam_size=5,
            word_timestamps=True,
        )
        segments = []
        all_words = []
        texts = []
        seg_idx = 0
        for seg in segments_iter:
            seg_words = []
            if seg.words:
                for w in seg.words:
                    word_dict = {
                        "word": w.word,
                        "start": round(w.start, 3),
                        "end": round(w.end, 3),
                        "probability": round(w.probability, 4),
                        "segment_idx": seg_idx,
                    }
                    seg_words.append(word_dict)
                    all_words.append(word_dict)
            segments.append({
                "text": seg.text,
                "start": seg.start,
                "end": seg.end,
                "avg_logprob": seg.avg_logprob,
                "no_speech_prob": seg.no_speech_prob,
                "n_words": len(seg_words),
            })
            texts.append(seg.text)
            seg_idx += 1

        hypothesis = "".join(texts).strip()
        n_segs = len(segments)
        avg_lp = (sum(s["avg_logprob"] for s in segments) / n_segs
                  if n_segs > 0 else 0.0)
        avg_ns = (sum(s["no_speech_prob"] for s in segments) / n_segs
                  if n_segs > 0 else 0.0)
        elapsed = time.time() - t0

        return {
            "hypothesis": hypothesis,
            "segments": segments,
            "words": all_words,
            "avg_logprob": round(avg_lp, 4),
            "no_speech_prob": round(avg_ns, 4),
            "duration": round(info.duration, 3) if info.duration else 0.0,
            "language": info.language,
            "language_prob": round(info.language_probability, 4),
            "inference_time": round(elapsed, 3),
        }
