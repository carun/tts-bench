"""Chatterbox (Resemble AI), original and Turbo.

Both are two-stage and autoregressive in the first stage, so they sit with
Magpie and Qwen on the cost axis rather than with Kokoro and FastPitch:

  * original -- `ResembleAI/chatterbox`. A 0.5B Llama backbone (T3) emits
    speech tokens, then S3Gen (a flow-matching decoder plus HiFiGAN-style
    vocoder) turns them into 24 kHz audio. Classifier-free guidance is on by
    default (cfg_weight=0.5), which doubles the T3 batch.
  * turbo -- `ResembleAI/chatterbox-turbo`. A 350M GPT-2 backbone, a one-step
    distilled decoder, and no CFG.

Both are run with the library's default generation parameters and their
built-in voice (`conds.pt` in the checkpoint), not a cloned reference. Every
other engine here is a stock voice at stock settings, and so is this one.

Two things inside the timed region that are part of the released pipeline and
are left in rather than carved out:

  * `punc_norm()` rewrites the input before tokenising -- `:` becomes `,`, an
    ellipsis becomes a comma, a missing full stop is appended. That matters
    for the `numbers` category: "3:45" reaches the model as "3,45".
  * Every output is watermarked with Resemble's Perth implicit watermarker
    before it is returned. It is not optional in `generate()`, so it is a cost
    any caller pays, and the RTF includes it.

Chatterbox samples (temperature 0.8), so like Magpie and Qwen a single run is
a draw; `TTS_BENCH_SEED` reseeds it and suffixes the result name.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, "/harness")
from runner import bench  # noqa: E402

VARIANTS = {
    "original": ("chatterbox", "ResembleAI/chatterbox", "0.5B"),
    "turbo": ("chatterbox-turbo", "ResembleAI/chatterbox-turbo", "350M"),
}


def make_load(variant):
    def load():
        if variant == "turbo":
            from chatterbox.tts_turbo import ChatterboxTurboTTS as cls
        else:
            from chatterbox.tts import ChatterboxTTS as cls
        return cls.from_pretrained(device="cuda")

    return load


def synth(model, text):
    wav = model.generate(text)
    a = wav.detach().cpu().numpy() if hasattr(wav, "detach") else np.asarray(wav)
    return np.asarray(a, dtype=np.float32).squeeze(), int(model.sr)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=sorted(VARIANTS), default="original")
    ap.add_argument("--corpus", default="/work/corpus.json")
    ap.add_argument("--outroot", default="/work/out")
    ap.add_argument("--resroot", default="/work/results")
    a = ap.parse_args()

    name, repo, params = VARIANTS[a.variant]
    _sfx = os.environ.get("TTS_BENCH_SEED")
    if _sfx and _sfx != "1234":
        name += f"-s{_sfx}"

    import importlib.metadata as md

    import torch

    bench(
        name,
        make_load(a.variant),
        synth,
        a.corpus,
        f"{a.outroot}/{name}",
        f"{a.resroot}/{name}.json",
        meta={
            "family": "Chatterbox",
            "models": repo,
            "params": params,
            "voice": "built-in conds.pt",
            "chatterbox_tts": md.version("chatterbox-tts"),
            "torch": torch.__version__,
            "install": "pip --no-deps; torch/torchaudio unpinned from 2.6.0, all other pins honoured",
        },
    )
