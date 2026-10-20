"""Candidate 18: how many bits does the INPUT state need? The FP32 model is left untouched; the ERA5 state fed
to it is rounded to k-bit uniform codes per field (per variable, level and time slice: min/max range, round to nearest,
dequantise). Two modes: "init" rounds only the initial condition (00/06 UTC history), "state" also rounds the state fed
back at every rollout step (as if forecasts were stored in k bits between steps). Scored like a card against the FP32
reference cache, so qerr is comparable with the weight/activation cards.

  AQ_MODEL=pretrained python scripts/input_bits.py --bits 16 12 10 8 6 4 --modes init state [--split dev --steps 2]
Outputs results*/input_b<k>_<mode>__<split>_h<steps>/{records.jsonl, summary.json}.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cards import harness as H          # noqa: E402
from cards.recipe import build          # noqa: E402


def read_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def qdq_uniform(x: torch.Tensor, bits: int) -> torch.Tensor:
    """Round each [..., H, W] field (all leading dims are separate fields) to 2^bits uniform levels over its min/max."""
    flat = x.reshape(-1, x.shape[-2] * x.shape[-1]).float()
    lo, hi = flat.amin(1, keepdim=True), flat.amax(1, keepdim=True)
    n = 2 ** bits - 1
    step = ((hi - lo) / n).clamp(min=1e-30)
    q = torch.round((flat - lo) / step).clamp(0, n)
    return (lo + q * step).reshape(x.shape).to(x.dtype)


def quantise_batch(batch, bits: int):
    return dataclasses.replace(batch,
                               surf_vars={k: qdq_uniform(v, bits) for k, v in batch.surf_vars.items()},
                               atmos_vars={k: qdq_uniform(v, bits) for k, v in batch.atmos_vars.items()})


def rollout(model, date: str, steps: int, bits: int, mode: str):
    H._paths()
    from cards.compat import prepare
    batch = quantise_batch(H.raw_batch_fast(date), bits)
    current = prepare(model, batch)
    preds = []
    with torch.inference_mode():
        for step in range(steps):
            H.STEP_IDX = step
            p = model(current)
            preds.append(p.to("cpu"))
            nxt = dataclasses.replace(
                p,
                surf_vars={k: torch.cat([current.surf_vars[k][:, 1:], v], 1) for k, v in p.surf_vars.items()},
                atmos_vars={k: torch.cat([current.atmos_vars[k][:, 1:], v], 1) for k, v in p.atmos_vars.items()})
            current = quantise_batch(nxt, bits) if mode == "state" else nxt
    return preds, batch.metadata.lat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", nargs="+", type=int, default=[16, 12, 10, 8, 6, 4])
    ap.add_argument("--modes", nargs="+", default=["init", "state"], choices=["init", "state"])
    ap.add_argument("--split", default="dev", choices=list(H.SPLITS))
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--max-dates", type=int, default=None)
    ap.add_argument("--results", default=None)
    a = ap.parse_args()
    res = Path(a.results) if a.results else ROOT / ("results" if H.MODEL == "pretrained" else f"results_{H.MODEL}")
    dates = H.SPLITS[a.split][: a.max_dates]
    cache = res / f"_ref_fp32_h{a.steps}"
    base = {}
    for key, nm in (("fp32", "fp32"), ("q0", "q0_sim")):
        p = res / f"{nm}__{a.split}_h{a.steps}" / "records.jsonl"
        if p.exists():
            base[key] = read_jsonl(p)
    model, _ = build(json.loads((ROOT / "recipes" / "fp32.json").read_text()))
    truth = H.Truth()
    for mode in a.modes:
        for bits in a.bits:
            name = f"input_b{bits}_{mode}"
            out = res / f"{name}__{a.split}_h{a.steps}"
            out.mkdir(parents=True, exist_ok=True)
            (out / "recipe.json").write_text(json.dumps({"name": name, "kind": "fp32", "input_bits": bits, "mode": mode}, indent=1))
            rec_path = out / "records.jsonl"
            done = {x["date"] for x in read_jsonl(rec_path)}
            with rec_path.open("a", encoding="utf-8") as f:
                for d in dates:
                    if d in done:
                        continue
                    t1 = time.time()
                    preds, lat = rollout(model, d, a.steps, bits, mode)
                    sec = time.time() - t1
                    ref = H.load_ref(cache, d, a.steps)
                    if ref is None:
                        sys.exit(f"no FP32 reference cache for {d}")
                    for row in H.score(preds, lat, truth, ref):
                        row.update(date=d, seconds=sec, peak_gib=torch.cuda.max_memory_allocated() / 2 ** 30)
                        f.write(json.dumps(row) + "\n")
                    f.flush()
                    del preds
                    H.free()
            recs = read_jsonl(rec_path)
            s = H.summarize(recs, base)
            s.update(name=name, split=a.split, steps=a.steps, input_bits=bits, mode=mode)
            (out / "summary.json").write_text(json.dumps(s, indent=1), encoding="utf-8")
            print(json.dumps({k: s.get(k) for k in ("name", "n_dates", "qerr_pct", "excess_pct")}), flush=True)


if __name__ == "__main__":
    main()
