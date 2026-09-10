"""Shared timing harness.

Every engine imports this, so the stopwatch is literally the same code in all
four runs. That matters more than it sounds: the two easiest ways to publish a
bogus TTS speed comparison are to time CUDA work without synchronising, and to
let one engine pay for kernel autotuning that another had already amortised.

  * `torch.cuda.synchronize()` brackets every measured region. CUDA launches
    are asynchronous, so `t1 - t0` around a generate() call measures how long it
    took to *queue* the work, not to do it. Engines that queue more eagerly
    would look faster for no reason.
  * WARMUP_RUNS synthesises are thrown away before anything is recorded.
    Magpie JIT-compiles Triton kernels on first call; Kokoro and FastPitch do
    not. Without a warmup the first sentence of one engine carries a
    compilation cost the others never pay, and it lands in whichever category
    happens to sort first.
  * Audio duration comes from the returned sample count, not from a file
    header, so a resample on write cannot quietly change the denominator of
    the RTF.

Peak GPU memory is `torch.cuda.max_memory_allocated`, reset after warmup. It
measures tensors this process allocated, which is the number you can compare
across engines -- not `nvidia-smi`, which on this unified-memory box reports
`[N/A]` for VRAM anyway.
"""

import json
import os
import statistics
import time

import numpy as np
import soundfile as sf

WARMUP_RUNS = 3
WARMUP_TEXT = "This sentence is only here to warm the kernels up."
SEED = int(os.environ.get("TTS_BENCH_SEED", "1234"))


def seed_everything(seed=SEED):
    """Pin the RNGs.

    Magpie and Qwen sample; the same sentence twice gave 69632 and then 67584
    samples of audio. Without a fixed seed the WER of an autoregressive engine
    is a draw from a distribution and any single run is unreproducible. Seeding
    does not make them deterministic engines -- it makes THIS run repeatable,
    which is a different and weaker claim. Run-to-run spread is measured
    separately rather than hidden behind the seed.
    """
    import random

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _sync():
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def _peak_mib():
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() // 1024 // 1024
    except Exception:
        pass
    return None


def _reset_peak():
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def bench(engine, load, synth, corpus_path, out_dir, results_path, meta=None):
    """Run one engine over the corpus.

    `load()` -> model. `synth(model, text)` -> (float32 mono ndarray, sample_rate).
    """
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(corpus_path) as fh:
        corpus = json.load(fh)

    seed_everything()
    _sync()
    t0 = time.perf_counter()
    model = load()
    _sync()
    load_seconds = time.perf_counter() - t0
    print(f"[{engine}] load {load_seconds:.2f}s", flush=True)

    for _ in range(WARMUP_RUNS):
        try:
            synth(model, WARMUP_TEXT)
        except Exception as e:  # a failed warmup is not fatal; the real runs report
            print(f"[{engine}] warmup failed: {type(e).__name__}: {e}", flush=True)
            break
    _sync()
    _reset_peak()
    seed_everything()  # again: warmup consumed RNG draws

    rows, failures = [], 0
    for item in corpus:
        _sync()
        t0 = time.perf_counter()
        try:
            audio, sr = synth(model, item["text"])
            _sync()
            dt = time.perf_counter() - t0
        except Exception as e:
            _sync()
            failures += 1
            print(f"[{engine}] FAIL {item['id']}: {type(e).__name__}: {e}", flush=True)
            rows.append({**item, "error": f"{type(e).__name__}: {e}"})
            continue

        audio = np.asarray(audio, dtype=np.float32).squeeze()
        dur = len(audio) / float(sr)
        wav = os.path.join(out_dir, f"{item['id']}.wav")
        sf.write(wav, audio, int(sr))
        rows.append(
            {
                **item,
                "wav": wav,
                "synth_seconds": round(dt, 4),
                "audio_seconds": round(dur, 4),
                "rtf": round(dt / dur, 4) if dur > 0 else None,
                "sample_rate": int(sr),
                "peak_abs": round(float(np.abs(audio).max()), 4) if len(audio) else 0.0,
            }
        )
        print(f"[{engine}] {item['id']} {dt:.2f}s -> {dur:.2f}s (rtf {dt/dur:.3f})", flush=True)

    ok = [r for r in rows if "rtf" in r and r["rtf"] is not None]
    summary = {
        "engine": engine,
        "meta": meta or {},
        "load_seconds": round(load_seconds, 3),
        "peak_gpu_mib": _peak_mib(),
        "n_total": len(rows),
        "n_ok": len(ok),
        "n_failed": failures,
        "rtf_median": round(statistics.median([r["rtf"] for r in ok]), 4) if ok else None,
        "rtf_mean": round(statistics.fmean([r["rtf"] for r in ok]), 4) if ok else None,
        "audio_seconds_total": round(sum(r["audio_seconds"] for r in ok), 2),
        "synth_seconds_total": round(sum(r["synth_seconds"] for r in ok), 2),
        "items": rows,
    }
    with open(results_path, "w") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print(
        f"[{engine}] done: {len(ok)}/{len(rows)} ok, median RTF {summary['rtf_median']}, "
        f"peak {summary['peak_gpu_mib']} MiB -> {results_path}",
        flush=True,
    )
    return summary
