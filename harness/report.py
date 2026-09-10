"""Aggregate engine results and judge scores into the tables the report uses.

Deliberately dumb: it joins JSON and prints markdown. All the judgement calls
live upstream in the corpus, the harness and the judges, which is where they can
be argued with.
"""

import argparse
import glob
import json
import os
import statistics

CATS = ["prose", "numbers", "acronyms", "jargon", "names", "complex", "heteronyms"]


def load_all(resdir, scoredir):
    engines = {}
    for p in sorted(glob.glob(os.path.join(resdir, "*.json"))):
        d = json.load(open(p))
        engines[d["engine"]] = {"run": d, "judges": {}}
    for p in sorted(glob.glob(os.path.join(scoredir, "*.json"))):
        d = json.load(open(p))
        if d["engine"] in engines:
            engines[d["engine"]]["judges"][d["judge"]] = d
    return engines


def fmt(x, nd=3, dash="—"):
    return dash if x is None else f"{x:.{nd}f}"


def pct(x, nd=1, dash="—"):
    return dash if x is None else f"{100*x:.{nd}f}%"


def speed_table(engines):
    rows = []
    for name, e in sorted(engines.items()):
        r = e["run"]
        ok = [i for i in r["items"] if i.get("rtf")]
        rtfs = [i["rtf"] for i in ok]
        rows.append(
            {
                "engine": name,
                "params": r["meta"].get("params", "—"),
                "load": r["load_seconds"],
                "rtf_med": r["rtf_median"],
                "rtf_p90": round(statistics.quantiles(rtfs, n=10)[8], 4) if len(rtfs) > 8 else None,
                "xrt": round(1 / r["rtf_median"], 1) if r["rtf_median"] else None,
                "mib": r["peak_gpu_mib"],
                "sr": r["meta"].get("native_sr", "—"),
                "audio": r["audio_seconds_total"],
                "wall": r["synth_seconds_total"],
                "failed": r["n_failed"],
            }
        )
    out = [
        "| Engine | Params | Load (s) | RTF median | RTF p90 | × real-time | Peak GPU | SR |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(rows, key=lambda x: (x["rtf_med"] is None, x["rtf_med"])):
        out.append(
            f"| {r['engine']} | {r['params']} | {fmt(r['load'],1)} | {fmt(r['rtf_med'],4)} | "
            f"{fmt(r['rtf_p90'],4)} | {r['xrt']}× | {r['mib']} MiB | {r['sr']} |"
        )
    return "\n".join(out), rows


def wer_table(engines, judge):
    out = ["| Engine | Overall WER | " + " | ".join(CATS) + " |", "|---" * (len(CATS) + 2) + "|"]
    rows = []
    for name, e in sorted(engines.items()):
        j = e["judges"].get(judge)
        if not j:
            continue
        cells = []
        for c in CATS:
            v = j["by_category"].get(c, {}).get("wer")
            cells.append(pct(v))
        rows.append((name, j["overall_wer"], cells))
    for name, overall, cells in sorted(rows, key=lambda x: (x[1] is None, x[1])):
        out.append(f"| {name} | **{pct(overall)}** | " + " | ".join(cells) + " |")
    return "\n".join(out)


def agreement_table(engines):
    out = ["| Engine | Parakeet WER | Whisper WER | Δ |", "|---|---|---|---|"]
    for name, e in sorted(engines.items()):
        p = e["judges"].get("parakeet", {}).get("overall_wer")
        w = e["judges"].get("whisper", {}).get("overall_wer")
        d = None if (p is None or w is None) else abs(p - w)
        out.append(f"| {name} | {pct(p)} | {pct(w)} | {pct(d)} |")
    return "\n".join(out)


def worst_items(engines, judge, n=12):
    rows = []
    for name, e in sorted(engines.items()):
        j = e["judges"].get(judge)
        if not j:
            continue
        for it in j["items"]:
            if it["wer"]:
                rows.append((it["wer"], name, it))
    rows.sort(key=lambda x: -x[0])
    out = ["| Engine | Cat | WER | Asked | Heard |", "|---|---|---|---|---|"]
    for w, name, it in rows[:n]:
        asked = it["text"].replace("|", "\\|")
        heard = (it["hyp"] or "").replace("|", "\\|")[:90]
        out.append(f"| {name} | {it['category']} | {pct(w)} | {asked} | {heard} |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--scored", default="scored")
    ap.add_argument("--out", default="REPORT_DATA.md")
    a = ap.parse_args()

    engines = load_all(a.results, a.scored)
    st, _ = speed_table(engines)
    parts = [
        "## Speed\n\n" + st,
        "\n## Intelligibility — Parakeet judge\n\n" + wer_table(engines, "parakeet"),
        "\n## Intelligibility — Whisper judge\n\n" + wer_table(engines, "whisper"),
        "\n## Judge agreement\n\n" + agreement_table(engines),
        "\n## Worst individual utterances (Whisper judge)\n\n" + worst_items(engines, "whisper"),
    ]
    text = "\n".join(parts) + "\n"
    with open(a.out, "w") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
