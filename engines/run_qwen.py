"""Qwen3-TTS 12Hz 0.6B CustomVoice.

The largest model here and the only one that is an LLM in the ordinary sense:
a Qwen backbone autoregressively emits audio codec tokens at 12Hz. Two
consequences the numbers will show:

  * Cost tracks generated tokens. A 12Hz token rate is the floor on how much it
    must emit per second of speech, so RTF has a hard structural component that
    a non-autoregressive vocoder like HiFiGAN simply does not pay.
  * It has no phoneme frontend at all. Text goes in as text and the model has
    learned pronunciation end to end, so there is no dictionary to miss and no
    grapheme fallback to trip over -- but equally no way to *correct* it when it
    gets a proper noun wrong. That trade is the interesting axis against
    FastPitch, which fails loudly and is fixable, and Kokoro, which predicts.

`attn_implementation` is "sdpa", not the "flash_attention_2" the model card
suggests: flash-attn publishes no aarch64 wheel, and building it on this box
takes longer than the entire benchmark. sdpa is PyTorch's own fused attention
and is the honest default here; it does not change output, only speed, and it
is noted in the report rather than hidden.

The `instruct` parameter (natural-language style control) is deliberately left
unset. Every other engine here is being asked for a plain neutral reading, and
prompting one of them for affect would make the comparison meaningless.
"""

import argparse
import sys

import numpy as np
import torch

sys.path.insert(0, "/harness")
from runner import bench  # noqa: E402

MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
SPEAKER = "Aiden"  # "Sunny American male voice"


def load():
    from qwen_tts import Qwen3TTSModel

    return Qwen3TTSModel.from_pretrained(
        MODEL,
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )


def synth(model, text):
    wavs, sr = model.generate_custom_voice(text=text, language="English", speaker=SPEAKER)
    a = wavs[0]
    a = a.detach().cpu().numpy() if hasattr(a, "detach") else np.asarray(a)
    return np.asarray(a, dtype=np.float32).squeeze(), int(sr)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="/work/corpus.json")
    ap.add_argument("--outroot", default="/work/out")
    ap.add_argument("--resroot", default="/work/results")
    a = ap.parse_args()

    import os
    _sfx = os.environ.get("TTS_BENCH_SEED")
    name = "qwen3-tts-0.6b" + (f"-s{_sfx}" if _sfx and _sfx != "1234" else "")

    bench(
        name,
        load,
        synth,
        a.corpus,
        f"{a.outroot}/{name}",
        f"{a.resroot}/{name}.json",
        meta={
            "family": "Qwen",
            "models": MODEL,
            "params": "0.6B",
            "speaker": SPEAKER,
            "dtype": "bfloat16",
            "attn": "sdpa",
        },
    )
