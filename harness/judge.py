"""Score intelligibility by transcribing synthesised audio and measuring WER.

There is no way to measure TTS "accuracy" directly without human listeners, so
this uses the standard proxy: synthesise text, transcribe the result, and count
how far the transcript drifted from what the engine was asked to say. It
measures intelligibility, not naturalness -- an engine can score perfectly and
still sound like a robot. That limitation is stated in the report rather than
papered over.

TWO judges, from different families, because one judge is a conflict of
interest. Parakeet is NVIDIA, and so are FastPitch and Magpie; scoring NVIDIA
synthesis with an NVIDIA recogniser trained on overlapping data invites exactly
the objection you would raise reading someone else's benchmark. Whisper is the
independent second opinion. Where they agree, the number is probably real.
Where they disagree, the disagreement is the finding, and both are reported.

Normalisation is the other place these benchmarks quietly go wrong. Whisper's
EnglishTextNormalizer is applied identically to reference and hypothesis: it
folds case and punctuation, expands contractions, and canonicalises numbers, so
"$1,247.50" and "one thousand two hundred forty seven dollars fifty" have a
chance of meeting in the middle. It is not perfect on currency and phone
numbers, which is why `numbers` is reported as its own category -- a residual
error rate there is partly the normaliser's and should not be read as an engine
defect without listening.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import soundfile as sf

TARGET_SR = 16000


def _load_16k(path):
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != TARGET_SR:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
    return np.ascontiguousarray(audio, dtype=np.float32)


def _normalizer():
    from whisper_normalizer.english import EnglishTextNormalizer

    return EnglishTextNormalizer()


def wer(ref_words, hyp_words):
    """Levenshtein over word lists. Returns (errors, ref_len, sub, ins, dele)."""
    n, m = len(ref_words), len(hyp_words)
    if n == 0:
        return m, 0, 0, m, 0
    # dp[i][j] = (cost, sub, ins, del)
    prev = [(j, 0, j, 0) for j in range(m + 1)]
    for i in range(1, n + 1):
        cur = [(i, 0, 0, i)] + [None] * m
        for j in range(1, m + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                cur[j] = prev[j - 1]
            else:
                c_sub = prev[j - 1][0] + 1
                c_del = prev[j][0] + 1
                c_ins = cur[j - 1][0] + 1
                best = min(c_sub, c_del, c_ins)
                if best == c_sub:
                    p = prev[j - 1]
                    cur[j] = (best, p[1] + 1, p[2], p[3])
                elif best == c_ins:
                    p = cur[j - 1]
                    cur[j] = (best, p[1], p[2] + 1, p[3])
                else:
                    p = prev[j]
                    cur[j] = (best, p[1], p[2], p[3] + 1)
        prev = cur
    cost, sub, ins, dele = prev[m]
    return cost, n, sub, ins, dele


def transcribe_parakeet(paths):
    from nemo.collections.asr.models import ASRModel

    m = ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v2").eval().to("cuda")
    out = m.transcribe(paths, batch_size=8)
    res = []
    for h in out:
        res.append(h.text if hasattr(h, "text") else str(h))
    del m
    return res


def transcribe_whisper(paths, model_id="openai/whisper-large-v3"):
    import torch
    from transformers import pipeline

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model_id,
        device="cuda:0",
        torch_dtype=torch.float16,
        chunk_length_s=30,
    )
    res = []
    for p in paths:
        r = pipe(_load_16k(p), return_timestamps=False)
        res.append(r["text"] if isinstance(r, dict) else str(r))
    del pipe
    return res


JUDGES = {"parakeet": transcribe_parakeet, "whisper": transcribe_whisper}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="engine results json from runner.bench")
    ap.add_argument("--judge", required=True, choices=sorted(JUDGES))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    with open(a.results) as fh:
        summary = json.load(fh)
    items = [r for r in summary["items"] if r.get("wav") and os.path.exists(r["wav"])]
    if not items:
        print(f"no audio for {summary['engine']}", file=sys.stderr)
        json.dump({"engine": summary["engine"], "judge": a.judge, "items": []}, open(a.out, "w"))
        return

    paths = [r["wav"] for r in items]
    print(f"[judge:{a.judge}] transcribing {len(paths)} files from {summary['engine']}", flush=True)
    hyps = JUDGES[a.judge](paths)

    norm = _normalizer()
    scored, tot_err, tot_ref = [], 0, 0
    for r, hyp in zip(items, hyps):
        ref_n = norm(r["text"])
        hyp_n = norm(hyp)
        rw, hw = ref_n.split(), hyp_n.split()
        err, reflen, sub, ins, dele = wer(rw, hw)
        tot_err += err
        tot_ref += reflen
        scored.append(
            {
                "id": r["id"],
                "category": r["category"],
                "text": r["text"],
                "hyp": hyp.strip(),
                "ref_norm": ref_n,
                "hyp_norm": hyp_n,
                "errors": err,
                "ref_len": reflen,
                "sub": sub,
                "ins": ins,
                "del": dele,
                "wer": round(err / reflen, 4) if reflen else None,
            }
        )

    by_cat = {}
    for s in scored:
        c = by_cat.setdefault(s["category"], {"errors": 0, "ref_len": 0, "n": 0})
        c["errors"] += s["errors"]
        c["ref_len"] += s["ref_len"]
        c["n"] += 1
    for c in by_cat.values():
        c["wer"] = round(c["errors"] / c["ref_len"], 4) if c["ref_len"] else None

    out = {
        "engine": summary["engine"],
        "judge": a.judge,
        "overall_wer": round(tot_err / tot_ref, 4) if tot_ref else None,
        "total_errors": tot_err,
        "total_ref_words": tot_ref,
        "by_category": by_cat,
        "items": scored,
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print(f"[judge:{a.judge}] {summary['engine']} WER {out['overall_wer']} -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
