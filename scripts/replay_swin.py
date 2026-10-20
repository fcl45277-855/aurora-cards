"""Replay diagnostic for the backbone's Swin window attention (B deep-dive): same measurement as
replay_attn.py but for every WindowAttention block, accumulated per stage (enc0..2, dec0..2).

K perturbation = Q0 activation-only NVFP4 on attn.qkv (SimNVFP4Linear: weight rotated in BF16, full-width Hadamard
with seed SEED + manifest index of the qkv layer, dynamic 16384-token chunk scales), taken on the K third of the qkv
output; the logit error dl = q . dk / sqrt(D) (+ the shifted-window mask, as in swin3d.WindowAttention.forward) is
injected at alpha in {1, 1/2, 1/4}; the model continues with the exact output. Per stage and date:
  flip rate, share of error in flipped cells, value gap of the swapped tokens (B score = flip * vgap), A, G, p12.

  AQ_PERCEIVER_CHUNK=16384 AQ_MEM_FRAC=0.57 python scripts/replay_swin.py --dates 2024-02-02 2024-02-06 2024-02-10
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch
from einops import rearrange

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cards import harness as H                      # noqa: E402
from cards.recipe import build                      # noqa: E402
from cards.rotation import BlockHadamard            # noqa: E402
from cards.simlinear import SimNVFP4Linear          # noqa: E402

ALPHAS = (1.0, 0.5, 0.25)
SUBW = 256                                           # windows per measurement sub-block


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dates", nargs="+", default=["2024-02-02", "2024-02-06", "2024-02-10"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if int(os.environ.get("AQ_PERCEIVER_CHUNK", "0")) == 0:
        sys.exit("set AQ_PERCEIVER_CHUNK=16384")
    frac = float(os.environ.get("AQ_MEM_FRAC", "0"))
    if frac:
        torch.cuda.set_per_process_memory_fraction(frac)
    if a.out is None:
        a.out = str(ROOT / ("results" if H.MODEL == "pretrained" else f"results_{H.MODEL}") / "_replay_swin.json")
    model, _ = build({"kind": "bf16"})
    H._paths()
    from aurora.model.swin3d import WindowAttention
    idx = {n: i for i, n in enumerate(H.manifest_layers(), 1)}
    mods = {n: m for n, m in model.named_modules() if isinstance(m, WindowAttention)}
    sims, stage_of = {}, {}
    for n, m in mods.items():
        w = m.qkv.weight.detach().float()
        rot = BlockHadamard(w.shape[1], w.shape[1], seed=H.SEED + idx[n + ".qkv"], device="cuda", compute_dtype=torch.bfloat16)
        sims[n] = SimNVFP4Linear(rot.rotate_weight(w.to(torch.bfloat16)).float(), torch.ones((), device="cuda"), None,
                                 rotation=rot, act_bits="nvfp4", rot_dtype=torch.bfloat16).cuda()
        p = n.split(".")
        stage_of[n] = ("enc" if p[1] == "encoder_layers" else "dec") + p[2]
    accs, cur = {}, {}

    def add(key, val):
        cur[key] = cur.get(key, 0.0) + float(val)

    @torch.no_grad()
    def measure(self, name, x, mask, rollout_step):
        st = name                      # accumulate per block; stages are aggregated in the summary
        Hh, D = self.num_heads, self.dim
        dim = D
        qkv_f = (self.qkv(x) + self.lora_qkv(x, rollout_step)).float()      # lora_qkv is a module (finetuned) or a 0-lambda
        dk_all = sims[name](x).float()[..., dim:2 * dim] - (x.float() @ self.qkv.weight.float().t())[..., dim:2 * dim]
        nW = x.shape[0]
        if mask is not None:
            mw = mask.float()                                       # (nWin, ws, ws), repeats over batch
        for s in range(0, nW, SUBW):
            qkv = qkv_f[s:s + SUBW]
            q, k, v = rearrange(qkv, "B N (t h d) -> t B h N d", t=3, h=Hh)
            dk = rearrange(dk_all[s:s + SUBW], "B N (h d) -> B h N d", h=Hh)
            scale = q.shape[-1] ** -0.5
            l0 = (q @ k.transpose(-1, -2)) * scale
            if mask is not None:
                l0 = l0 + mw[(torch.arange(s, s + q.shape[0], device=x.device) % mw.shape[0])].unsqueeze(1)
            dl = (q @ dk.transpose(-1, -2)) * scale
            p0 = l0.softmax(-1)
            big = p0.max(-1).values >= 0.8
            srt = l0.sort(-1, descending=True)
            add(f"{st}:cells", big.numel()); add(f"{st}:big", big.sum())
            add(f"{st}:gap", (srt.values[..., 0] - srt.values[..., 1]).sum())
            add(f"{st}:relK_num", (dk ** 2).sum()); add(f"{st}:relK_den", (k ** 2).sum())
            add(f"{st}:cdl2", ((dl - dl.mean(-1, keepdim=True)) ** 2).sum())
            add(f"{st}:cl02", ((l0 - l0.mean(-1, keepdim=True)) ** 2).sum())
            for al in ALPHAS:
                d = al * dl
                ec = (((l0 + d).softmax(-1) - p0) @ v).pow(2).sum(-1)
                el = ((p0 * (d - (p0 * d).sum(-1, keepdim=True))) @ v).pow(2).sum(-1)
                flip = (l0 + d).argmax(-1) != l0.argmax(-1)
                add(f"{st}:E2_{al}", ec.sum()); add(f"{st}:El2_{al}", el.sum())
                add(f"{st}:E2flip_{al}", (ec * flip).sum()); add(f"{st}:flip_{al}", flip.sum())
                add(f"{st}:E2big_{al}", (ec * big).sum())
                if al == 1.0:
                    i1 = srt.indices[..., 0:1].unsqueeze(-1).expand(-1, -1, -1, 1, v.shape[-1])
                    i2 = srt.indices[..., 1:2].unsqueeze(-1).expand(-1, -1, -1, 1, v.shape[-1])
                    vexp = v.unsqueeze(2).expand(-1, -1, l0.shape[2], -1, -1)
                    vgap2 = (vexp.gather(3, i1) - vexp.gather(3, i2)).squeeze(3).pow(2).sum(-1)
                    add(f"{st}:vgap2_flip", (vgap2 * flip).sum()); add(f"{st}:vgap2_all", vgap2.sum())
                    add(f"{st}:vnorm2", v.pow(2).sum(-1).mean(-1, keepdim=True).expand_as(vgap2).sum())

    orig = WindowAttention.forward
    name_of = {m: n for n, m in mods.items()}

    def fwd(self, x, mask=None, rollout_step=0):
        measure(self, name_of[self], x, mask, rollout_step)
        return orig(self, x, mask=mask, rollout_step=rollout_step)

    WindowAttention.forward = fwd
    try:
        for d in a.dates:
            cur = {}
            H.rollout(model, d, 1, False)
            accs[d] = cur
            print(f"[{H.now()}] {d} done", flush=True)
    finally:
        WindowAttention.forward = orig

    def summ(c, st):
        g = lambda k: c[f"{st}:{k}"]
        rel_k = math.sqrt(g("relK_num") / g("relK_den"))
        r = {"rel_K": rel_k, "A": math.sqrt(g("cdl2") / g("cl02")) / rel_k, "share_big_cells": g("big") / g("cells"),
             "mean_gap": g("gap") / g("cells"),
             "vgap_rel_flipped": (g("vgap2_flip") / max(g("flip_1.0"), 1)) ** 0.5 / (g("vnorm2") / g("cells")) ** 0.5,
             "vgap_rel_all": (g("vgap2_all") / g("cells")) ** 0.5 / (g("vnorm2") / g("cells")) ** 0.5}
        for al in ALPHAS:
            r[f"alpha_{al}"] = {"G": math.sqrt(g(f"E2_{al}") / g(f"El2_{al}")), "flip_rate": g(f"flip_{al}") / g("cells"),
                                "share_err_in_flipped": g(f"E2flip_{al}") / max(g(f"E2_{al}"), 1e-30),
                                "share_big_all": g(f"E2big_{al}") / max(g(f"E2_{al}"), 1e-30)}
        r["p12"] = 0.5 * math.log2(g("E2_1.0") / g("E2_0.5")); r["p24"] = 0.5 * math.log2(g("E2_0.5") / g("E2_0.25"))
        r["B_score"] = r["alpha_1.0"]["flip_rate"] * r["vgap_rel_flipped"]
        return r

    stages = sorted({v for v in stage_of.values()})
    def stage_sums(c, stg):
        agg = {}
        for n, s in stage_of.items():
            if s != stg:
                continue
            for k, v in c.items():
                if k.startswith(n + ":"):
                    kk = stg + ":" + k.split(":", 1)[1]
                    agg[kk] = agg.get(kk, 0.0) + v
        return agg
    res = {"model": H.MODEL,
           "dates": {d: {stg: summ(stage_sums(c, stg), stg) for stg in stages} for d, c in accs.items()},
           "blocks": {d: {n: summ(c, n) for n in mods} for d, c in accs.items()},
           "stage_of": stage_of, "peak_gib": torch.cuda.max_memory_allocated() / 2 ** 30}
    Path(a.out).write_text(json.dumps(res, indent=1))
    for st in stages:
        s = res["dates"][a.dates[0]][st]
        print(f"{st}: relK {s['rel_K']:.3f} A {s['A']:.1f} G {s['alpha_1.0']['G']:.2f} p12 {s['p12']:.2f} flip {s['alpha_1.0']['flip_rate']:.4f} "
              f"err_in_flip {s['alpha_1.0']['share_err_in_flipped']:.3f} vgap {s['vgap_rel_flipped']:.3f} B {s['B_score']:.4f}")


if __name__ == "__main__":
    main()
