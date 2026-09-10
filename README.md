# tts-bench

Intelligibility and speed benchmark for four open text-to-speech engines, run
on an NVIDIA GB10 (Grace Blackwell, arm64, compute capability 12.1).

[Four open TTS engines on a GB10](https://arunc.dev/essays/four-open-tts-engines-on-a-gb10/).

| Engine | Params | RTF (median) | Peak GPU | WER (Whisper judge) |
|---|---|---|---|---|
| [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) | 82M | 0.0126 | 707 MiB | **2.3%** |
| [Qwen3-TTS 0.6B](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice) | 0.6B | 0.588 | 2301 MiB | 4.1% |
| [MagpieTTS 357M](https://huggingface.co/nvidia/magpie_tts_multilingual_357m) | 357M | 0.412 | 2345 MiB | 7.0% |
| NeMo FastPitch + HiFiGAN | 140M | 0.0085 | 729 MiB | 7.8% |

Chatterbox was excluded: it pins `torch==2.6.0`, which has no Blackwell kernels.

## How it works

56 sentences across 7 categories (prose, numbers, acronyms, jargon, names,
complex clauses, heteronyms) are synthesised by each engine. Two independent
ASR systems then transcribe every output, and word error rate is computed
against the input text after identical normalisation of both sides.

Two judges rather than one because a single judge is a conflict of interest:
Parakeet is NVIDIA's recogniser and two of the engines under test are NVIDIA's
synthesisers. Whisper large-v3 is the outside opinion. Both are reported.

### What it measures, and what it does not

WER here is a proxy for **intelligibility** — whether a machine can recover the
words. It is not naturalness. The two diverge, and this benchmark contains a
worked example of them diverging: the `fastpitch-fixed` variant is audibly
better than `fastpitch-default` and scores 1.6 points *worse*.

The `heteronyms` category measures nothing at all. "read" transcribes as "read"
whichever way it was pronounced, so every engine scores 0%. It is retained in
the output for honesty, not for information. Scoring it properly needs
phoneme-level forced alignment or human raters.

## Running it

Requires podman with the NVIDIA CDI hook, and a CUDA GPU.

```bash
./scripts/build.sh        # three engine images over one shared torch base
./scripts/run_all.sh      # synthesise the corpus with every engine
./scripts/judge_all.sh    # transcribe and score, both judges
./scripts/variance.sh     # optional: reseed the two sampling engines
python3 harness/report.py --results data/results --scored data/scored
```

`BENCH_WORK` (default `~/tts-bench-work`) and `HF_CACHE` (default
`~/.cache/tts-bench-hf`) override the working and weights directories. The
first run downloads roughly 8 GB of model weights.

## Layout

```
harness/corpus.py      the 56 sentences, and why each category exists
harness/runner.py      shared stopwatch — cuda sync, warmup, seeding
harness/judge.py       ASR transcription, normalisation, WER
harness/report.py      joins the JSON into markdown tables
engines/run_*.py       one adapter per engine, common interface
containers/*           one image per irreconcilable dependency set
data/                  raw per-utterance results from the published run
```

Each engine adapter supplies `load()` and `synth(model, text)`; everything else
— timing, warmup, seeding, output — is `harness/runner.py`, so the stopwatch is
literally the same code for all of them.

## Notes for anyone reproducing this on arm64

* MagpieTTS builds the G2P for all twelve of its languages at load, so an
  English-only run still needs `jieba`, `pypinyin`, `pypinyin_dict` and
  `pyopenjtalk`. A missing one surfaces as an `UnboundLocalError` from inside
  hydra, naming a variable rather than the module.
* `pyopenjtalk` is source-only on PyPI. `pyopenjtalk-plus` ships an aarch64
  wheel under the same import name.
* Magpie JIT-compiles Triton kernels during inference, so `gcc` must be present
  at **runtime**. On a slim image it loads fine and then fails with
  `RuntimeError: Failed to find C compiler`.
* torch reports `sm_120` as its highest arch while the GB10 reports capability
  `(12, 1)`. It runs regardless, via minor-version compatibility.
* flash-attn has no arm64 wheel, so Qwen runs with `sdpa` attention.
