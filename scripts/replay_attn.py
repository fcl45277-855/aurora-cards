"""Replay diagnostic for the decoder to_kv mechanism (reduced to the K path).

On the BF16 (autocast) model, the decoder's level_decoder cross-attention is re-run with keys from the m1kv_act
perturbation (activation-only NVFP4 on to_kv through SimNVFP4Linear: weight rotated in BF16, activation FWHT in the
input dtype = FP32 in the decoder under autocast, dynamic 16384-token chunk scales; dk = sim(x)[K] - to_kv_fp32(x)[K],
which also contains the BF16 GEMM rounding, as in the m1kv_* cards). The logit error dl = q . dk / 8 is injected at
amplitude alpha in {1, 1/2, 1/4}. The model itself continues with the exact output (pure measurement).

Per date (06 requires each rule on each of the 3 dates) it reports, on the attention output before to_out:
  G   = ||E(1)|| / ||E_lin(1)||  (E_lin: first-order softmax), overall and at the 500 / 850 hPa queries
  p12 = log2(||E(1)|| / ||E(1/2)||), p24 likewise, overall and at 500 / 850 hPa
  share_big_500 = share of the 500 hPa-query error carried by cells with p_top1 >= 0.8 (M_big)
  flip rates (argmax change), top-1/top-2 swaps, A = [rms(centred dl) / rms(centred logits)] / rel_K
Proxy verdict (not the full planned test: no K_common / +-15% gates, errors are attention-output errors, not z500 MSE):
  H2-proxy: G >= 1.8, p12 >= 1.3, share_big_500(1) >= 0.5 and share_big_500(1/4) < 0.5 * share_big_500(1)
  H1-proxy: G in [0.85, 1.15], p12 <= 1.15, A >= 3 (06 also needs |I_K| / (K_a + K_w) <= 0.2: pending)
  else "generic softmax curvature" if G or p12 >= 1.3; a final label only if all dates agree.

  AQ_PERCEIVER_CHUNK=16384 AQ_MEM_FRAC=0.57 python scripts/replay_attn.py --dates 2024-02-02 2024-02-06 2024-02-10
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
NAMES = {"decoder": "decoder.level_decoder.layers.0.0.to_kv", "encoder": "encoder.level_agg.layers.0.0.to_kv"}
SUB = 2048                                           # patches per measurement sub-block (memory)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dates", nargs="+", default=["2024-02-02", "2024-02-06", "2024-02-10"])
    ap.add_argument("--module", default="decoder", choices=list(NAMES))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if int(os.environ.get("AQ_PERCEIVER_CHUNK", "0")) == 0:
        sys.exit("set AQ_PERCEIVER_CHUNK=16384 (memory; keeps the 16384-token activation-scale chunks aligned)")
    frac = float(os.environ.get("AQ_MEM_FRAC", "0"))
    if frac:
        torch.cuda.set_per_process_memory_fraction(frac)
    NAME = NAMES[a.module]
    if a.out is None:
        a.out = str(ROOT / ("results" if H.MODEL == "pretrained" else f"results_{H.MODEL}") / f"_replay_attn_{a.module}.json")
    model, _ = build({"kind": "bf16"})
    att = (model.decoder.level_decoder if a.module == "decoder" else model.encoder.level_agg).layers[0][0]
    # decoder queries = the 13 pressure levels; encoder queries = latent levels (no physical label): report overall only
    levs = {"500": H.LEVELS.index(500), "850": H.LEVELS.index(850)} if a.module == "decoder" else {}
    w = att.to_kv.weight.detach().float()
    k_in = w.shape[1]
    idx = {n: i for i, n in enumerate(H.manifest_layers(), 1)}[NAME]
    rot = BlockHadamard(k_in, k_in, seed=H.SEED + idx, device="cuda", compute_dtype=torch.bfloat16)
    sim = SimNVFP4Linear(rot.rotate_weight(w.to(torch.bfloat16)).float(), torch.ones((), device="cuda"), None,
                         rotation=rot, act_bits="nvfp4", rot_dtype=torch.bfloat16).cuda()
    accs, cur = {}, {}

    def add(key, val):
        cur[key] = cur.get(key, 0.0) + float(val)

    @torch.no_grad()
    def measure(self, latents, x):
        h, inner = self.num_heads, self.inner_dim
        dk_all = sim(x).float()[..., :inner] - self.to_kv(x).float()[..., :inner]   # whole call: same scale chunks
        rs = lambda t: rearrange(t, "b l (h d) -> b h l d", h=h)
        for s in range(0, latents.shape[0], SUB):
            lat, xs, dk = latents[s:s + SUB], x[s:s + SUB], dk_all[s:s + SUB]
            q = self.to_q(lat).float()
            k, v = self.to_kv(xs).float().chunk(2, dim=-1)
            qh, kh, vh, dkh = rs(q), rs(k), rs(v), rs(dk)
            scale = qh.shape[-1] ** -0.5
            l0 = (qh @ kh.transpose(-1, -2)) * scale                 # (b, h, L1 queries, L2 keys)
            dl = (qh @ dkh.transpose(-1, -2)) * scale
            p0 = l0.softmax(-1)
            big = p0.max(-1).values >= 0.8
            srt = l0.sort(-1, descending=True)
            gap = srt.values[..., 0] - srt.values[..., 1]
            dgap = (dl.gather(-1, srt.indices[..., 1:2]) - dl.gather(-1, srt.indices[..., 0:1])).squeeze(-1)
            add("cells", big.numel())
            add("big", big.sum())
            add("gap_sum", gap.sum())
            add("dgap2", dgap.square().sum())
            add("margin_noise_le1", (gap <= dgap.abs()).sum())
            add("logit_elements", l0.numel())
            add("output2", (p0 @ vh).square().sum())
            add("swap12_1", (dgap > gap).sum())
            add("relK_num", (dk ** 2).sum())
            add("relK_den", (k ** 2).sum())
            add("cdl2", ((dl - dl.mean(-1, keepdim=True)) ** 2).sum())
            add("cl02", ((l0 - l0.mean(-1, keepdim=True)) ** 2).sum())
            for al in ALPHAS:
                d = al * dl
                ec = (((l0 + d).softmax(-1) - p0) @ vh).pow(2).sum(-1)          # per cell (b, h, L1)
                el = ((p0 * (d - (p0 * d).sum(-1, keepdim=True))) @ vh).pow(2).sum(-1)
                add(f"E2_{al}", ec.sum())
                add(f"El2_{al}", el.sum())
                for tag, li in levs.items():
                    add(f"E2_{tag}_{al}", ec[:, :, li].sum())
                    add(f"El2_{tag}_{al}", el[:, :, li].sum())
                add(f"E2big_all_{al}", (ec * big).sum())
                if "500" in levs:
                    add(f"E2big_500_{al}", (ec[:, :, levs["500"]] * big[:, :, levs["500"]]).sum())
                flip = (l0 + d).argmax(-1) != l0.argmax(-1)
                add(f"flip_{al}", flip.sum())
                add(f"flipbig_{al}", (flip & big).sum())
                # direction B: error carried by switched vs non-switched cells, and the value gap of the
                # two swapped tokens (||v_top1 - v_top2|| per cell) on switched cells
                add(f"E2flip_{al}", (ec * flip).sum())
                if al == 1.0:
                    i1 = srt.indices[..., 0:1].unsqueeze(-1).expand(-1, -1, -1, 1, vh.shape[-1])
                    i2 = srt.indices[..., 1:2].unsqueeze(-1).expand(-1, -1, -1, 1, vh.shape[-1])
                    vexp = vh.unsqueeze(2).expand(-1, -1, l0.shape[2], -1, -1)      # (b, h, L1, L2, d)
                    vgap2 = (vexp.gather(3, i1) - vexp.gather(3, i2)).squeeze(3).pow(2).sum(-1)   # (b, h, L1)
                    add("vgap2_flip", (vgap2 * flip).sum()); add("vgap2_all", vgap2.sum())
                    add("vnorm2_all", vh.pow(2).sum(-1).mean(-1, keepdim=True).expand_as(vgap2).sum())

    cls = type(att)
    orig = cls.forward

    def fwd(self, latents, x):
        if self is att:
            measure(self, latents, x)             # locals freed before the real attention runs
        return orig(self, latents, x)

    cls.forward = fwd
    try:
        for d in a.dates:
            cur = {}
            H.rollout(model, d, 1, False)
            accs[d] = cur
            print(f"[{H.now()}] {d} done", flush=True)
    finally:
        cls.forward = orig

    def summarise(c):
        rel_k = math.sqrt(c["relK_num"] / c["relK_den"])
        pair_noise_rms = math.sqrt(c["dgap2"] / c["cells"])
        r = {"mean_gap_over_pair_noise_rms": (c["gap_sum"] / c["cells"]) / max(pair_noise_rms, 1e-30),
             "pair_noise_rms": pair_noise_rms,
             "fraction_margin_le_abs_pair_noise": c["margin_noise_le1"] / c["cells"],
             "centered_dlogit_rms": math.sqrt(c["cdl2"] / c["logit_elements"]),
             "centered_logit_rms": math.sqrt(c["cl02"] / c["logit_elements"]),
             "raw_accumulators": c,
             "share_big_cells": c["big"] / c["cells"], "mean_gap": c["gap_sum"] / c["cells"],
             "top12_swap_rate_alpha1": c["swap12_1"] / c["cells"], "rel_K": rel_k,
             "A": math.sqrt(c["cdl2"] / c["cl02"]) / rel_k}
        for al in ALPHAS:
            r[f"alpha_{al}"] = {"relative_output_error": math.sqrt(c[f"E2_{al}"] / max(c["output2"], 1e-30)),
                                "G": math.sqrt(c[f"E2_{al}"] / c[f"El2_{al}"]),
                                **{f"G_{t}": math.sqrt(c[f"E2_{t}_{al}"] / c[f"El2_{t}_{al}"]) for t in levs},
                                "share_big_all": c[f"E2big_all_{al}"] / max(c[f"E2_{al}"], 1e-30),
                                **({"share_big_500": c[f"E2big_500_{al}"] / max(c[f"E2_500_{al}"], 1e-30)} if "500" in levs else {}),
                                "flip_rate": c[f"flip_{al}"] / c["cells"],
                                "flip_rate_big": c[f"flipbig_{al}"] / max(c["big"], 1),
                                "share_err_in_flipped": c[f"E2flip_{al}"] / max(c[f"E2_{al}"], 1e-30)}
        r["vgap_rel_flipped"] = float((c["vgap2_flip"] / max(c["flip_1.0"], 1)) ** 0.5 / (c["vnorm2_all"] / c["cells"]) ** 0.5)
        r["vgap_rel_all"] = float((c["vgap2_all"] / c["cells"]) ** 0.5 / (c["vnorm2_all"] / c["cells"]) ** 0.5)
        for t in [""] + [f"_{k}" for k in levs]:
            key = lambda al: c[f"E2{t}_{al}"]
            r[f"p12{t}"] = 0.5 * math.log2(key(1.0) / key(0.5))
            r[f"p24{t}"] = 0.5 * math.log2(key(0.5) / key(0.25))
        g, p12 = r["alpha_1.0"]["G"], r["p12"]
        sk = "share_big_500" if "500" in levs else "share_big_all"
        sb1, sb4 = r["alpha_1.0"][sk], r["alpha_0.25"][sk]
        if g >= 1.8 and p12 >= 1.3 and sb1 >= 0.5 and sb4 < 0.5 * sb1:
            r["proxy_verdict"] = "H2-proxy (nonlinear switching)"
        elif 0.85 <= g <= 1.15 and p12 <= 1.15 and r["A"] >= 3:
            r["proxy_verdict"] = "H1-proxy (linear common-mode; I_K gate pending)"
        elif g >= 1.3 or p12 >= 1.3:
            r["proxy_verdict"] = "generic softmax curvature"
        else:
            r["proxy_verdict"] = "none of H1/H2"
        return r

    res = {"module": a.module, "model": H.MODEL, "layer": NAME, "dates": {d: summarise(c) for d, c in accs.items()},
           "env": {"AQ_PERCEIVER_CHUNK": os.environ.get("AQ_PERCEIVER_CHUNK"), "AQ_MEM_FRAC": frac},
           "peak_gib": torch.cuda.max_memory_allocated() / 2 ** 30}
    verdicts = {r["proxy_verdict"] for r in res["dates"].values()}
    res["final_proxy_verdict"] = verdicts.pop() if len(verdicts) == 1 else "dates disagree"
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
