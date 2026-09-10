"""Kokoro-82M.

The small one, and the reason the parameter count is worth watching in the
results: 82M against Magpie's 357M and Qwen's 600M. Kokoro is a StyleTTS2-family
non-autoregressive model, so like FastPitch its cost tracks input length rather
than generated tokens.

Its frontend is misaki, which uses a dictionary plus espeak-ng as fallback for
out-of-vocabulary words. That fallback is the structural difference from
FastPitch: an unknown word gets *predicted* phonemes rather than being dropped
to letters. Whether prediction beats spelling is an empirical question, which
is what the `jargon` and `names` categories are for.

`KPipeline` yields per-segment results because it chunks on `split_pattern`.
Corpus sentences are single-line so it normally yields once, but the segments
are concatenated rather than assumed, because a sentence long enough to split
would otherwise be silently truncated to its first clause.
"""

import argparse
import sys

import numpy as np

sys.path.insert(0, "/harness")
from runner import bench  # noqa: E402

VOICE = "af_heart"
LANG = "a"  # American English
SR = 24000


def load():
    from kokoro import KPipeline

    return KPipeline(lang_code=LANG)


def _chunk_audio(item):
    """Tolerate both the tuple API and the newer Result object."""
    if hasattr(item, "audio"):
        return item.audio
    if isinstance(item, (tuple, list)):
        return item[-1]
    return item


def synth(model, text):
    parts = []
    for item in model(text, voice=VOICE, speed=1):
        a = _chunk_audio(item)
        if a is None:
            continue
        a = a.detach().cpu().numpy() if hasattr(a, "detach") else np.asarray(a)
        parts.append(np.asarray(a, dtype=np.float32).squeeze())
    if not parts:
        raise RuntimeError("kokoro produced no audio segments")
    return (parts[0] if len(parts) == 1 else np.concatenate(parts)), SR


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="/work/corpus.json")
    ap.add_argument("--outroot", default="/work/out")
    ap.add_argument("--resroot", default="/work/results")
    a = ap.parse_args()

    bench(
        "kokoro",
        load,
        synth,
        a.corpus,
        f"{a.outroot}/kokoro",
        f"{a.resroot}/kokoro.json",
        meta={
            "family": "Kokoro",
            "models": "hexgrad/Kokoro-82M",
            "params": "82M",
            "voice": VOICE,
            "native_sr": SR,
        },
    )
