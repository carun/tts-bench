"""NeMo FastPitch + HiFiGAN.

Two variants, and the comparison between them is half the reason this engine is
in the benchmark:

  default  exactly what you get from `FastPitchModel.from_pretrained(...)`
  fixed    the same weights with `ignore_ambiguous_words = False` and a small
           pronunciation dictionary for out-of-vocabulary terms

`EnglishG2p` defaults to `ignore_ambiguous_words=True`, which drops any word
with more than one CMUdict pronunciation to raw *letters* rather than phonemes.
That rule catches "the", "is", "on", "and" -- four of the commonest words in
English. Fed letters, the model pronounces letters: "sidecar" comes out
"side-e-car", silent 'e' voiced. Mixed into properly phonemised words, the
result reads to a listener as a foreign accent rather than as a bug, which is
exactly why it survives in so much sample audio.

Note the dictionary keys are LOWERCASE. Writing "SIDECAR" adds an entry that is
never looked up, no error, and the word still spells out.
"""

import argparse
import sys

import torch

sys.path.insert(0, "/harness")
from runner import bench  # noqa: E402

# Out-of-vocabulary terms in the corpus, as ARPAbet. Kept deliberately short:
# this is the maintenance burden a phoneme-dictionary frontend imposes, and
# padding it would hide that cost.
OOV = {
    "sidecar": ["S", "AY1", "D", "K", "AA2", "R"],
    "diarization": ["D", "AY2", "ER0", "AH0", "Z", "EY1", "SH", "AH0", "N"],
    "kubernetes": ["K", "UW2", "B", "ER0", "N", "EH1", "T", "IY0", "Z"],
    "quantisation": ["K", "W", "AA2", "N", "T", "AH0", "Z", "EY1", "SH", "AH0", "N"],
    "backpropagation": ["B", "AE2", "K", "P", "R", "AA2", "P", "AH0", "G", "EY1", "SH", "AH0", "N"],
    "tokenizer": ["T", "OW1", "K", "AH0", "N", "AY2", "Z", "ER0"],
    "graphemes": ["G", "R", "AE1", "F", "IY0", "M", "Z"],
    "phonemes": ["F", "OW1", "N", "IY0", "M", "Z"],
    "autoscaler": ["AO1", "T", "OW0", "S", "K", "EY2", "L", "ER0"],
    "embeddings": ["EH0", "M", "B", "EH1", "D", "IH0", "NG", "Z"],
}


def build(variant):
    def load():
        from nemo.collections.tts.models import FastPitchModel, HifiGanModel

        spec = FastPitchModel.from_pretrained("tts_en_fastpitch")
        if variant == "fixed":
            tok = spec.vocab
            g2p = getattr(tok, "g2p", None)
            for obj in (tok, g2p):
                if obj is not None and getattr(obj, "phoneme_probability", None) is not None:
                    obj.phoneme_probability = 1.0
            if g2p is not None:
                g2p.ignore_ambiguous_words = False
                for word, phones in OOV.items():
                    g2p.phoneme_dict[word] = [phones]
        spec = spec.eval().to("cuda")
        voc = HifiGanModel.from_pretrained("tts_en_hifigan").eval().to("cuda")
        return spec, voc

    def synth(model, text):
        spec, voc = model
        with torch.no_grad():
            mel = spec.generate_spectrogram(tokens=spec.parse(text))
            audio = voc.convert_spectrogram_to_audio(spec=mel)
        return audio.detach().to("cpu").float().numpy().squeeze(), 22050

    return load, synth


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["default", "fixed"], default="fixed")
    ap.add_argument("--corpus", default="/work/corpus.json")
    ap.add_argument("--outroot", default="/work/out")
    ap.add_argument("--resroot", default="/work/results")
    a = ap.parse_args()

    name = f"fastpitch-{a.variant}"
    load, synth = build(a.variant)
    bench(
        name,
        load,
        synth,
        a.corpus,
        f"{a.outroot}/{name}",
        f"{a.resroot}/{name}.json",
        meta={
            "family": "NeMo",
            "models": "tts_en_fastpitch + tts_en_hifigan",
            "params": "~55M (FastPitch) + ~85M (HiFiGAN)",
            "variant": a.variant,
            "native_sr": 22050,
        },
    )
