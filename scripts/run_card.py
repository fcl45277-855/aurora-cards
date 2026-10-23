"""Run one card (recipe) on a date split and score it.

  python scripts/run_card.py --recipe recipes/fp32.json --split dev        # first: builds the FP32 reference cache
  python scripts/run_card.py --recipe recipes/native_q0.json --split dev   # Q0 baseline
  python scripts/run_card.py --recipe recipes/gptq_full.json --split dev

Outputs results/<name>__<split>_h<steps>/{recipe.json, build_info.json, records.jsonl, summary.json}.
Resumable per date: a date counts as done only when all its records (steps x variables) are present; records of
a partly written date are dropped and the date is run again. A malformed line in records.jsonl other than an
unterminated last line (an interrupted write) stops the run. The FP32 card writes the reference cache results/_ref_fp32_h<steps>/ used for rmse_ref.
Summaries pair each card with results/fp32__<split>_h<steps> and native_q0__<split>_h<steps> when present.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch  # noqa: E402

from cards import harness as H  # noqa: E402
from cards.recipe import build, merged  # noqa: E402


def read_jsonl(p: Path) -> list[dict]:
    """Records of a JSONL file. Only an unterminated last line (a write interrupted by a crash) is skipped, with a
    message; any other malformed line raises."""
    if not p.exists():
        return []
    text = p.read_text(encoding="utf-8")
    lines = text.split("\n")
    out = []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as e:
            if i == len(lines) - 1:                       # no trailing newline: the interrupted last write
                print(f"{p}: ignoring the incomplete last line", flush=True)
                continue
            raise ValueError(f"{p}: malformed line {i + 1}: {e}") from None
    return out


def complete_dates(recs: list[dict], steps: int) -> set[str]:
    """Dates with exactly one record per (step, variable)."""
    seen = {}
    for x in recs:
        seen.setdefault(x["date"], set()).add((x["step"], x["var"]))
    need = {(s, v) for s in range(1, steps + 1) for v in H.VARS}
    return {d for d, k in seen.items() if k == need and sum(1 for x in recs if x["date"] == d) == len(need)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", required=True)
    ap.add_argument("--split", default="dev", choices=list(H.SPLITS))
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--results", default=str(ROOT / ("results" if H.MODEL == "pretrained" else f"results_{H.MODEL}")))
    ap.add_argument("--max-dates", type=int, default=None)
    a = ap.parse_args()
    recipe = json.loads(Path(a.recipe).read_text(encoding="utf-8"))
    r = merged(recipe)
    res = Path(a.results)
    res.mkdir(parents=True, exist_ok=True)
    marker = res / "_model.json"              # one model per results dir (FP32 reference cache and baselines live here)
    tag = {"model": H.MODEL, "ckpt": H.CKPT.name}
    if marker.exists():
        if json.loads(marker.read_text()) != tag:
            sys.exit(f"{res} belongs to {marker.read_text().strip()}, not {tag}: pass the matching --results")
    else:
        marker.write_text(json.dumps(tag))
    out = res / f"{r['name']}__{a.split}_h{a.steps}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "recipe.json").write_text(json.dumps(r, indent=1), encoding="utf-8")
    dates = H.SPLITS[a.split][: a.max_dates]
    rec_path = out / "records.jsonl"
    old = read_jsonl(rec_path)
    done = complete_dates(old, a.steps)
    partial = sorted({x["date"] for x in old} - done)
    if partial or (old and not rec_path.read_text(encoding="utf-8").endswith("\n")):
        print(f"dropping the records of partly written dates {partial} and rewriting {rec_path}", flush=True)
        rec_path.write_text("".join(json.dumps(x) + "\n" for x in old if x["date"] in done), encoding="utf-8")
    todo = [d for d in dates if d not in done]
    cache = res / f"_ref_fp32_h{a.steps}"
    is_ref = r["kind"] == "fp32"
    if todo:
        frac = float(os.environ.get("AQ_MEM_FRAC", "0"))
        if frac:            # cap this process (a shared GPU): this process runs out of memory before squeezing other jobs
            torch.cuda.set_per_process_memory_fraction(frac)
        t0 = time.time()
        model, info = build(recipe)
        info["build_seconds"] = time.time() - t0
        info["perceiver_chunk"] = int(os.environ.get("AQ_PERCEIVER_CHUNK", "0"))
        info["mem_frac"] = frac
        (out / "build_info.json").write_text(json.dumps(info, indent=1, default=str), encoding="utf-8")
        truth = H.Truth()
        torch.cuda.reset_peak_memory_stats()
        with rec_path.open("a", encoding="utf-8") as f:
            for d in todo:
                t1 = time.time()
                preds, lat = H.rollout(model, d, a.steps, r["surface_fp32"] and not is_ref)
                sec = time.time() - t1
                if is_ref:
                    H.save_ref(cache, d, preds)
                ref = H.load_ref(cache, d, a.steps)
                for row in H.score(preds, lat, truth, ref):
                    row.update(date=d, seconds=sec, peak_gib=torch.cuda.max_memory_allocated() / 2 ** 30)
                    f.write(json.dumps(row) + "\n")
                f.flush()
                print(f"[{H.now()}] {r['name']} {d}: {sec:.1f}s", flush=True)
                del preds
                H.free()
    recs = read_jsonl(rec_path)
    base = {}
    for key, nm in (("fp32", "fp32"), ("q0", "q0_sim")):        # since 2026-10-18 Q0 baseline = q0_sim (same regime)
        p = res / f"{nm}__{a.split}_h{a.steps}" / "records.jsonl"
        if p.exists() and nm != r["name"]:
            base[key] = read_jsonl(p)
    s = H.summarize(recs, base)
    s.update(name=r["name"], split=a.split, steps=a.steps,
             mean_seconds_per_date=sum(x["seconds"] for x in recs) / max(1, len(recs)),
             peak_gib=max((x["peak_gib"] for x in recs), default=None))
    (out / "summary.json").write_text(json.dumps(s, indent=1), encoding="utf-8")
    print(json.dumps({k: s.get(k) for k in ("name", "n_dates", "qerr_pct", "qerr_reduction_vs_q0_pct",
                                              "qerr_reduction_vs_q0_pct_ci95", "excess_pct", "excess_pct_ci95",
                                              "mean_seconds_per_date", "peak_gib")}), flush=True)


if __name__ == "__main__":
    main()
