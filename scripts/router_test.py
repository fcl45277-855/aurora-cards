"""Candidate 15: router hypothesis. In the FP32 model, replace the softmax of Aurora's learned-query Perceiver
attention (encoder.level_agg and/or decoder.level_decoder) by a hard argmax (one-hot) or a top-k renormalised softmax
and score the rollout like a card (qerr vs the FP32 reference cache). The "softmax" mode is the control (same
attention computed by hand; qerr should be ~0) and records how peaked the FP32 attention is per module: mean
max-probability, entropy (nats, and normalised by log L2), share of cells with max-prob > 0.9, max-prob histogram.

  AQ_MODEL=pretrained python scripts/router_test.py --modes softmax argmax top2 --modules enc dec both [--split dev --steps 2]

Outputs results*/router_<mode>_<modules>__<split>_h<steps>/ (card-like: records.jsonl, summary.json) and
results*/_router_stats_<split>_h<steps>.json (attention peakedness, softmax mode).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cards import harness as H          # noqa: E402
from cards.recipe import build          # noqa: E402

MODULES = {"enc": ("encoder.level_agg",), "dec": ("decoder.level_decoder",),
           "both": ("encoder.level_agg", "decoder.level_decoder")}


def read_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


class Patch:
    """Swap PerceiverAttention.forward for the duration of one mode; modules not in `prefixes` keep the original."""

    def __init__(self, model, prefixes, mode: str, stats: dict | None):
        from aurora.model.perceiver import PerceiverAttention
        self.cls = PerceiverAttention
        self.orig = PerceiverAttention.__dict__.get("_aq_orig_forward", PerceiverAttention.forward)
        PerceiverAttention._aq_orig_forward = self.orig
        self.tagged = []
        for name, m in model.named_modules():
            if isinstance(m, PerceiverAttention):
                if any(name.startswith(p) for p in prefixes):
                    m._aq_mode, m._aq_name, m._aq_stats = mode, name, stats
                    self.tagged.append(m)
        orig = self.orig

        def forward(self_, latents, x):
            if not hasattr(self_, "_aq_mode"):
                return orig(self_, latents, x)
            from einops import rearrange
            h = self_.num_heads
            q = self_.to_q(latents)
            k, v = self_.to_kv(x).chunk(2, dim=-1)
            k = self_.ln_k(k)
            q = self_.ln_q(q)
            q, k, v = map(lambda t: rearrange(t, "b l (h d) -> b h l d", h=h), (q, k, v))
            logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.shape[-1])      # (B, H, L1, L2)
            st = self_._aq_stats
            if st is not None:
                p = logits.float().softmax(-1)
                pm = p.amax(-1)
                s = st.setdefault(self_._aq_name, {"n": 0, "sum_maxp": 0.0, "sum_ent": 0.0, "n_gt09": 0, "n_gt099": 0,
                                                    "L2": int(p.shape[-1]), "heads": int(p.shape[1]), "L1": int(p.shape[2]),
                                                    "hist_maxp": [0] * 10})
                s["n"] += pm.numel()
                s["sum_maxp"] += float(pm.sum())
                s["sum_ent"] += float(-(p * p.clamp_min(1e-12).log()).sum(-1).sum())
                s["n_gt09"] += int((pm > 0.9).sum())
                s["n_gt099"] += int((pm > 0.99).sum())
                s["hist_maxp"] = [a + int(b) for a, b in zip(s["hist_maxp"], torch.histc(pm, bins=10, min=0.0, max=1.0).tolist())]
            mode = self_._aq_mode
            if mode == "softmax":
                attn = logits.float().softmax(-1)
            elif mode == "argmax":
                attn = F.one_hot(logits.argmax(-1), logits.shape[-1]).float()
            elif mode.startswith("top"):
                kk = int(mode[3:])
                idx = logits.topk(min(kk, logits.shape[-1]), -1).indices
                mask = torch.full_like(logits, float("-inf")).scatter(-1, idx, 0.0)
                attn = (logits + mask).float().softmax(-1)
            else:
                raise ValueError(mode)
            out = torch.matmul(attn.to(v.dtype), v)
            out = rearrange(out, "B H L1 D -> B L1 (H D)")
            return self_.to_out(out)

        PerceiverAttention.forward = forward

    def close(self):
        self.cls.forward = self.orig
        for m in self.tagged:
            for a in ("_aq_mode", "_aq_name", "_aq_stats"):
                if hasattr(m, a):
                    delattr(m, a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=["softmax", "argmax", "top2"])
    ap.add_argument("--modules", nargs="+", default=["enc", "dec", "both"], choices=list(MODULES))
    ap.add_argument("--split", default="dev", choices=list(H.SPLITS))
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--max-dates", type=int, default=None)
    ap.add_argument("--results", default=None)
    a = ap.parse_args()
    res = Path(a.results) if a.results else ROOT / ("results" if H.MODEL == "pretrained" else f"results_{H.MODEL}")
    tag = {"model": H.MODEL, "ckpt": H.CKPT.name}
    marker = res / "_model.json"
    if marker.exists() and json.loads(marker.read_text()) != tag:
        sys.exit(f"{res} belongs to another model")
    dates = H.SPLITS[a.split][: a.max_dates]
    cache = res / f"_ref_fp32_h{a.steps}"
    base = {}
    for key, nm in (("fp32", "fp32"), ("q0", "q0_sim")):
        p = res / f"{nm}__{a.split}_h{a.steps}" / "records.jsonl"
        if p.exists():
            base[key] = read_jsonl(p)
    model, _ = build(json.loads((ROOT / "recipes" / "fp32.json").read_text()))
    truth = H.Truth()
    stats_all = {}
    for mods in a.modules:
        for mode in a.modes:
            if mode == "softmax" and mods != "both" and "both" in a.modules:
                continue                                    # the control only needs one run (stats cover both modules)
            name = f"router_{mode}_{mods}"
            out = res / f"{name}__{a.split}_h{a.steps}"
            out.mkdir(parents=True, exist_ok=True)
            (out / "recipe.json").write_text(json.dumps({"name": name, "kind": "fp32", "router_mode": mode, "modules": mods}, indent=1))
            rec_path = out / "records.jsonl"
            done = {x["date"] for x in read_jsonl(rec_path)}
            stats = {} if mode == "softmax" else None
            patch = Patch(model, MODULES[mods], mode, stats)
            try:
                with rec_path.open("a", encoding="utf-8") as f:
                    for d in dates:
                        if d in done:
                            continue
                        t1 = time.time()
                        preds, lat = H.rollout(model, d, a.steps, False)
                        sec = time.time() - t1
                        ref = H.load_ref(cache, d, a.steps)
                        if ref is None:
                            sys.exit(f"no FP32 reference cache for {d} in {cache}")
                        for row in H.score(preds, lat, truth, ref):
                            row.update(date=d, seconds=sec, peak_gib=torch.cuda.max_memory_allocated() / 2 ** 30)
                            f.write(json.dumps(row) + "\n")
                        f.flush()
                        print(f"[{H.now()}] {name} {d}: {sec:.1f}s", flush=True)
                        del preds
                        H.free()
            finally:
                patch.close()
            recs = read_jsonl(rec_path)
            s = H.summarize(recs, base)
            s.update(name=name, split=a.split, steps=a.steps, router_mode=mode, modules=mods)
            (out / "summary.json").write_text(json.dumps(s, indent=1), encoding="utf-8")
            print(json.dumps({k: s.get(k) for k in ("name", "n_dates", "qerr_pct", "excess_pct")}), flush=True)
            if stats:
                for n, v in stats.items():
                    stats_all[n] = {**v, "mean_maxp": v["sum_maxp"] / v["n"], "mean_entropy": v["sum_ent"] / v["n"],
                                    "entropy_over_log_L2": v["sum_ent"] / v["n"] / math.log(v["L2"]),
                                    "share_maxp_gt09": v["n_gt09"] / v["n"], "share_maxp_gt099": v["n_gt099"] / v["n"]}
                    print(n, {k: round(stats_all[n][k], 4) for k in ("mean_maxp", "entropy_over_log_L2", "share_maxp_gt09", "share_maxp_gt099")},
                          "L2", v["L2"], flush=True)
    if stats_all:
        (res / f"_router_stats_{a.split}_h{a.steps}.json").write_text(json.dumps({"model": H.MODEL, "dates": dates, "modules": stats_all}, indent=1))


if __name__ == "__main__":
    main()
