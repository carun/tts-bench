"""NVIDIA MagpieTTS Multilingual 357M.

Autoregressive: a transformer encoder-decoder predicts discrete audio codec
tokens which NanoCodec decodes to waveform. That architecture is why its RTF
behaves differently from FastPitch's -- cost scales with the number of tokens
it decides to emit, not with the length of the input text, so a sentence it
"decides" to say slowly costs more.

Two operational surprises worth knowing before you deploy it:

  * It builds the G2P for ALL twelve supported languages at load, so an
    English-only deployment still needs jieba, pypinyin, pypinyin_dict and
    pyopenjtalk. Miss one and the load fails inside hydra with an
    UnboundLocalError naming a variable, not a module.
  * Inference JIT-compiles Triton kernels, so a C compiler must be present at
    RUNTIME. On a slim base image the failure is
    `RuntimeError: Failed to find C compiler` from deep inside triton, long
    after the model has loaded successfully.

speaker_index picks one of five English voices (Aria, Jason, John, Leo, Sofia).
Sofia (4) is used here; the choice is arbitrary and held constant.
"""

import argparse
import sys

sys.path.insert(0, "/harness")
from runner import bench  # noqa: E402

SPEAKER = 4  # Sofia
APPLY_TN = False  # set by --tn


def load():
    from nemo.collections.tts.models import MagpieTTSModel

    m = MagpieTTSModel.from_pretrained("nvidia/magpie_tts_multilingual_357m")
    return m.eval().to("cuda")


def synth(model, text):
    audio, _alen = model.do_tts(text, language="en", apply_TN=APPLY_TN, speaker_index=SPEAKER)
    sr = int(getattr(model, "sample_rate", 22050) or 22050)
    return audio.detach().to("cpu").float().numpy().squeeze(), sr


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="/work/corpus.json")
    ap.add_argument("--outroot", default="/work/out")
    ap.add_argument("--resroot", default="/work/results")
    ap.add_argument("--tn", action="store_true", help="enable NeMo text normalisation")
    a = ap.parse_args()

    APPLY_TN = a.tn
    import os
    _sfx = os.environ.get("TTS_BENCH_SEED")
    name = "magpie-tn" if a.tn else "magpie"
    if _sfx and _sfx != "1234":
        name += f"-s{_sfx}"

    bench(
        name,
        load,
        synth,
        a.corpus,
        f"{a.outroot}/{name}",
        f"{a.resroot}/{name}.json",
        meta={
            "family": "NeMo",
            "models": "nvidia/magpie_tts_multilingual_357m",
            "params": "357M + NanoCodec",
            "speaker": "Sofia",
            "apply_TN": a.tn,
            "native_sr": 22050,
        },
    )
