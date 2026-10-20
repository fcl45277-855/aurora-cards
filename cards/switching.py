"""Switching-dominance diagnostic of attention under a key perturbation (paper Sect. 4.2; replay analysis).

Given the queries, keys and values of an attention module and the perturbation of its keys caused by quantisation
(dK = K_quantised - K_exact), the diagnostic replays the attention exactly and reports:

  flip_rate      share of attention cells (query, head) whose argmax key changes
  err_in_flip    share of the attention-output squared error carried by the switched cells  <- the predictor
  A              common-mode amplification: relative error of the centred logits / relative key error
  G              nonlinearity: exact output error / its first-order (linearised softmax) prediction
  share_sharp    share of cells whose top-1 probability is >= `sharp`
  rel_K          ||dK|| / ||K||

The paper ranks attention modules by err_in_flip and finds Spearman +0.95 / +0.98 / +0.90 against their measured
sensitivity on three Aurora variants, whereas the local activation error does not predict it. Inputs may have any
leading dimensions: q [..., Lq, d], k, dk, v [..., Lk, d]; heads go into the leading dimensions.
"""
from __future__ import annotations

import math

import torch


@torch.no_grad()
def switching_diagnostic(q: torch.Tensor, k: torch.Tensor, dk: torch.Tensor, v: torch.Tensor, *, alpha: float = 1.0,
                         sharp: float = 0.8) -> dict:
    q, k, dk, v = (t.float() for t in (q, k, dk, v))
    scale = q.shape[-1] ** -0.5
    l0 = (q @ k.transpose(-1, -2)) * scale                       # [..., Lq, Lk]
    dl = alpha * (q @ dk.transpose(-1, -2)) * scale
    p0 = l0.softmax(-1)
    out0 = p0 @ v
    ec = (((l0 + dl).softmax(-1) - p0) @ v).pow(2).sum(-1)       # exact squared output error per cell
    el = ((p0 * (dl - (p0 * dl).sum(-1, keepdim=True))) @ v).pow(2).sum(-1)   # linearised softmax
    flip = (l0 + dl).argmax(-1) != l0.argmax(-1)
    rel_k = math.sqrt(float((alpha * dk).pow(2).sum()) / max(float(k.pow(2).sum()), 1e-30))
    cdl = dl - dl.mean(-1, keepdim=True)
    cl0 = l0 - l0.mean(-1, keepdim=True)
    e_tot = float(ec.sum())
    return {
        "flip_rate": float(flip.float().mean()),
        "err_in_flip": float((ec * flip).sum()) / max(e_tot, 1e-30),
        "A": math.sqrt(float(cdl.pow(2).sum()) / max(float(cl0.pow(2).sum()), 1e-30)) / max(rel_k, 1e-30),
        "G": math.sqrt(e_tot / max(float(el.sum()), 1e-30)),
        "share_sharp": float((p0.max(-1).values >= sharp).float().mean()),
        "rel_K": rel_k,
        "relative_output_error": math.sqrt(e_tot / max(float(out0.pow(2).sum()), 1e-30)),
    }
