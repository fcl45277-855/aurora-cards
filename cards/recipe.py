"""Build an Aurora variant from a recipe dict (JSON). Keys (defaults reproduce Q0 numerics as closely as
the simulation allows):

  name           str
  kind           "fp32" | "bf16" | "native_q0" | "sim"        (sim = simulated NVFP4 on `layers`)
  layers         "q0" (195 layers of layer_ab_ck_refined) | "all314" | path to a list file
  precision      "autocast" (default since 2026-10-18: FP32 weights/IO, Aurora's official autocast runs the backbone in
                 BF16) | "bf16_cast" (old regime: whole model .to(bf16); raw physical inputs/outputs rounded to BF16)
  surface_fp32   false (Q0's FP32 surface-input patch; only meaningful with precision = "bf16_cast")
  rotation       {"block": "full" | 16 | 32 | 64 | 128 | null, "dtype": "bf16" | "fp32"}
  weight         {"method": "rtn" | "gptq" | "none", "percdamp": 0.01, "scale_mult": 1.0, "source": "fp32" | "bf16"}
                 (none = BF16 weights, only the activations are quantized)
                 weight.gptq_skip: [substrings]  layers that use RTN even when method = gptq
  act            "nvfp4" | "fp8" (optimistic per-16 FP32 scale) | "mxfp8" | "fp8_tensor" | "bf16"
  scale_min      "tiny" (default, TorchAO: block scale >= 2^-6) | "subnormal" (block scale >= 2^-9, E4M3 subnormals)
  kv_split       {"pattern", "quant_rows": [a, b] | "K" | "V", "alpha"} or a list of such: only those output rows carry
                 alpha x quantization error (K/V = first/second half of to_kv's rows)
  k_centre       [substrings]: K half of the output (first half of to_kv rows) computed from the input centred over
                 the context tokens (dim -2); exact for attention keys
  act_overrides  [[substring, "bf16"|"fp8"|"nvfp4"], ...]  per-layer activation precision, first match wins
  act_scale_search null | [1.0, 1.5]   per-block activation scale choice (4/6) by squared error
  act_step_mults  null | [1.0, 1.5]   NVFP4 activation block-scale multiplier chosen by rollout step (overrides search)
  weight.scale_search null | [1.0, 1.5] | wider list   per-block weight scale choice (RTN only)
  lowpass        null | {"pool": 4, "patterns": ["mlp.fc1", "mlp.fc2"]}   N1: BF16 coarse (block-mean) branch + NVFP4 residual
                 lowpass.mode: "mean" (N1) | "global" (HadaNorm-style) | "haar" (STaMP-style, 2-level Haar);
                 lowpass.coarse_bits: "bf16" | "fp8" (precision of the kept coarse / low band)
  low_rank       null | {"rank": 32}   SVDQuant-style BF16 low-rank branch on the smoothed weight; residual quantized
  smooth         null | {"alpha": 0.5}     SmoothQuant-style s_j = amax_x^a / amax_w^(1-a), before rotation
  calib          {"split": "calib", "steps": 2, "budget_gib": 2.5}   only used by gptq / smooth
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from . import harness as H
from . import nvfp4 as NV
from .gptq import gptq_nvfp4, proxy_error, transform_hessian
from .rotation import BlockHadamard
from .simlinear import SimNVFP4Linear, rtn_weight

DEFAULT = {"kind": "sim", "layers": "q0", "surface_fp32": False, "precision": "autocast", "rotation": {"block": "full", "dtype": "bf16"},
           "weight": {"method": "rtn", "percdamp": 0.01, "scale_mult": 1.0, "source": "bf16"}, "act": "nvfp4",
           "smooth": None, "act_scale_search": None, "act_overrides": [], "calib": {"split": "calib", "steps": 2, "budget_gib": 2.5}}


def merged(recipe: dict) -> dict:
    r = json.loads(json.dumps(DEFAULT))
    for k, v in recipe.items():
        if isinstance(v, dict) and isinstance(r.get(k), dict):
            r[k].update(v)
        else:
            r[k] = v
    return r


@torch.no_grad()
def collect_stats(model, names: list[str], dates: list[str], steps: int, surface_fp32: bool,
                  budget_gib: float, want_hessian: bool = True) -> dict[str, dict]:
    """Second moments C = sum x x^T (fp32) and per-channel amax of each target layer's input, from rollouts
    of the (unquantized) model. Layers are processed in passes whose Hessians fit `budget_gib` of GPU memory."""
    mods = dict(model.named_modules())
    stats: dict[str, dict] = {}
    sizes = {n: mods[n].in_features ** 2 * 4 / 2 ** 30 for n in names}
    passes, cur, used = [], [], 0.0
    for n in names:
        s = sizes[n] if want_hessian else 0.0
        if cur and used + s > budget_gib:
            passes.append(cur)
            cur, used = [], 0.0
        cur.append(n)
        used += s
    if cur:
        passes.append(cur)
    for pi, group in enumerate(passes):
        acc = {}
        handles = []

        def make(n):
            def hook(_m, inp):
                x = inp[0].reshape(-1, inp[0].shape[-1])
                a = acc.setdefault(n, {"amax": torch.zeros(x.shape[1], device=x.device), "n": 0,
                                       "C": torch.zeros(x.shape[1], x.shape[1], device=x.device) if want_hessian else None})
                a["amax"] = torch.maximum(a["amax"], x.abs().amax(0).float())
                a["n"] += x.shape[0]
                if want_hessian:
                    for i in range(0, x.shape[0], 16384):
                        xc = x[i:i + 16384].float()
                        a["C"].addmm_(xc.t(), xc)
            return hook
        for n in group:
            handles.append(mods[n].register_forward_pre_hook(make(n)))
        try:
            for d in dates:
                H.rollout(model, d, steps, surface_fp32)
        finally:
            for h in handles:
                h.remove()
        for n, a in acc.items():
            stats[n] = {"amax": a["amax"].cpu(), "n": a["n"], "C": None if a["C"] is None else a["C"].cpu()}
        del acc
        H.free()
        print(f"[{H.now()}] stats pass {pi + 1}/{len(passes)}: {len(group)} layers", flush=True)
    return stats


def build(recipe: dict, device: str = "cuda"):
    r = merged(recipe)
    info = {"recipe": r, "model": H.MODEL, "ckpt": str(H.CKPT)}
    NV.SCALE_MIN = NV.E4M3_SUBNORMAL_MIN if r.get("scale_min") == "subnormal" else NV.E4M3_TINY
    if r["kind"] == "native_q0":
        raise NotImplementedError("native_q0 loaded a packaged model of an earlier internal study and is not part of the "
                                  "release; the on-device results of the paper use scripts/native_w4a16.py and native_t7.py")
    model = H.load_fp32(device)
    heads = ("decoder.atmos_heads", "decoder.surf_heads")
    lin = {n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and not n.startswith(heads)}
    if lin != set(H.manifest_layers()):        # catches a manifest that does not belong to the loaded checkpoint
        raise RuntimeError(f"manifest does not match the loaded {H.MODEL} model ({len(lin)} non-head Linear layers)")
    if r["kind"] == "fp32":
        return model, info
    names = H.layer_set(r["layers"]) if r["kind"] == "sim" else []
    mods = dict(model.named_modules())
    w32 = {n: mods[n].weight.detach().float().cpu() for n in names}
    if r.get("precision", "autocast") == "autocast":
        model.autocast = True            # Aurora's own mode: backbone in BF16, encoder/decoder and IO in FP32
    else:
        model = model.to(torch.bfloat16)
    info["precision"] = r.get("precision", "autocast")
    if r["kind"] == "bf16":
        return model, info
    idx = {n: i for i, n in enumerate(H.manifest_layers(), 1)}
    need = r["weight"]["method"] == "gptq" or r["smooth"]
    stats = {}
    if need:
        c = r["calib"]
        stats = collect_stats(model, names, H.SPLITS[c["split"]], c["steps"], r["surface_fp32"], c["budget_gib"],
                              want_hessian=r["weight"]["method"] == "gptq")
    mods = dict(model.named_modules())
    blk = r["rotation"]["block"]
    rot_dtype = torch.float32 if r["rotation"].get("dtype") == "fp32" else torch.bfloat16
    layer_info = {}
    for n in names:
        lin = mods[n]
        K = lin.in_features
        w = w32[n].to(device)
        if r["weight"].get("source", "fp32") == "bf16":
            w = w.to(torch.bfloat16).float()
        smooth = None
        if r["smooth"]:
            a = float(r["smooth"]["alpha"])
            ax = stats[n]["amax"].to(device).clamp(min=1e-5)
            aw = w.abs().amax(0).clamp(min=1e-5)
            smooth = (ax ** a / aw ** (1 - a)).clamp(min=1e-5)
            w = w * smooth[None, :]
        low_rank = None
        lr_cfg = r.get("low_rank")
        if lr_cfg and lr_cfg.get("rank"):
            # SVDQuant-style: split the (smoothed) weight into a BF16 rank-r branch and a residual that is quantized.
            # Branch on unquantized activations: y_lr = (x / s) @ (S_r V_r^T)^T @ U_r^T, folded into lr_a, lr_b.
            rk = int(lr_cfg["rank"])
            U, S, Vh = torch.linalg.svd(w, full_matrices=False)
            L = (U[:, :rk] * S[:rk]) @ Vh[:rk]
            w = w - L
            a_mat = S[:rk, None] * Vh[:rk]
            if smooth is not None:
                a_mat = a_mat / smooth[None, :]
            low_rank = (a_mat, U[:, :rk])
        rot = None
        if blk:
            rot = BlockHadamard(K, K if blk == "full" else int(blk), seed=H.SEED + idx[n], device=device,
                                compute_dtype=rot_dtype)
            w = rot.rotate_weight(w.to(rot_dtype)).float()
        use_gptq = r["weight"]["method"] == "gptq" and not any(pat in n for pat in r["weight"].get("gptq_skip", []))
        if use_gptq:
            h = transform_hessian(stats[n]["C"].to(device), rot, smooth)
            parts, pts, gi = gptq_nvfp4(w, h, percdamp=r["weight"]["percdamp"], scale_mult=r["weight"]["scale_mult"])
            p_rtn, pts_rtn = rtn_weight(w, r["weight"]["scale_mult"])
            gi["proxy_gptq"] = proxy_error(w, parts * pts, h)
            gi["proxy_rtn"] = proxy_error(w, p_rtn * pts_rtn, h)
            layer_info[n] = gi
            del h
        elif r["weight"]["method"] == "none":       # activation-only attribution: BF16 weights (what autocast uses)
            parts, pts = w, torch.ones((), device=w.device)
        else:
            parts, pts = rtn_weight(w, r["weight"]["scale_mult"], r["weight"].get("scale_search"))
        act = next((a for pat, a in r.get("act_overrides", []) if pat in n), r["act"])
        parent, child = H.parent_of(model, n)
        setattr(parent, child, SimNVFP4Linear(parts, pts, lin.bias, rotation=rot, smooth=smooth,
                                              act_bits=act, rot_dtype=rot_dtype, low_rank=low_rank,
                                              act_scale_search=r.get("act_scale_search")))
        if r.get("act_step_mults"):
            getattr(parent, child).act_step_mults = [float(x) for x in r["act_step_mults"]]
        del w
    ks = r.get("kv_split")
    for spec in ([ks] if isinstance(ks, dict) else (ks or [])):        # one dict or a list of dicts
        for n in names:
            if spec["pattern"] in n:
                m = dict(model.named_modules())[n]
                m.w_exact = w32[n].to(device)
                qr = spec["quant_rows"]                                   # [a, b] | "K" (first half) | "V" (second half)
                half = m.out_features // 2
                m.quant_rows = (0, half) if qr == "K" else ((half, m.out_features) if qr == "V" else tuple(qr))
                m.err_alpha = float(spec.get("alpha", 1.0))
    for n in names:
        if any(p in n for p in r.get("k_centre") or []):
            m = dict(model.named_modules())[n]
            m.k_centre = (0, m.out_features // 2)          # to_kv: k, v = chunk(2) -> K = first half of the rows
    lp = r.get("lowpass")
    if lp and lp.get("pool"):
        n_set = 0
        for n in names:
            if any(pat in n for pat in lp.get("patterns", ["mlp.fc1", "mlp.fc2"])):
                m = dict(model.named_modules())[n]
                if isinstance(m, SimNVFP4Linear):
                    m.lowpass_pool = int(lp["pool"])
                    m.lowpass_mode = lp.get("mode", "mean")
                    m.coarse_bits = lp.get("coarse_bits", "bf16")
                    n_set += 1

        def grid_hook(block, args, kwargs):
            res = args[2] if len(args) > 2 else kwargs.get("res")
            for lin in (getattr(block.mlp, "fc1", None), getattr(block.mlp, "fc2", None)):
                if isinstance(lin, SimNVFP4Linear):
                    lin.grid = tuple(int(v) for v in res)
        n_blocks = 0
        for mod in model.modules():
            if hasattr(mod, "mlp") and hasattr(mod, "attn") and hasattr(mod.mlp, "fc1"):
                mod.register_forward_pre_hook(grid_hook, with_kwargs=True)
                n_blocks += 1
        aw = lp.get("attn_window")                 # e.g. [6, 12]: attention inputs are windows of (2, Wh, Ww) tokens
        n_attn = 0
        if aw:
            for n in names:
                if any(pat in n for pat in lp.get("attn_patterns", ["attn.qkv", "attn.proj"])):
                    m = dict(model.named_modules())[n]
                    if isinstance(m, SimNVFP4Linear):
                        m.lowpass_pool = int(lp.get("attn_pool", 2))
                        m.lowpass_mode = lp.get("mode", "mean")
                        m.coarse_bits = lp.get("coarse_bits", "bf16")
                        m.grid = (-1, int(aw[0]), int(aw[1]))
                        n_attn += 1
        info["lowpass"] = {"layers": n_set, "blocks_hooked": n_blocks, "attn_layers": n_attn}
    if layer_info:
        info["layers"] = layer_info
        g = sum(v["proxy_gptq"] for v in layer_info.values())
        t = sum(v["proxy_rtn"] for v in layer_info.values())
        info["proxy_total"] = {"gptq": g, "rtn": t, "ratio": g / t if t else None}
    H.free()
    return model.eval(), info
