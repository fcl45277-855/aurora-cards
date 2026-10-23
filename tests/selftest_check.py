"""Self-tests that need no Aurora weights (CPU ok; CUDA tests run when a GPU is visible).

  python scripts/selftest.py            # prints PASS/FAIL lines, exits non-zero on failure
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch  # noqa: E402

from cards.gptq import gptq_nvfp4, proxy_error, transform_hessian  # noqa: E402
from cards.nvfp4 import e2m1_round, qdq_blocks  # noqa: E402
from cards.rotation import BlockHadamard  # noqa: E402
from cards.simlinear import SimNVFP4Linear, rtn_weight  # noqa: E402

FAIL = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info, flush=True)
    if not ok:
        FAIL.append(name)


def main():
    g = torch.tensor([0.0, 0.25, 0.26, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 5.1, 7.0, -0.75, -5.0])
    check("e2m1 RNE", e2m1_round(g.clamp(-6, 6)).tolist() == [0, 0, 0.5, 1, 1, 2, 2, 4, 4, 6, 6, -1, -4],
          e2m1_round(g.clamp(-6, 6)).tolist())
    try:
        from cards.nvfp4 import check_against_torchao
        r = check_against_torchao()
        check("bit-exact vs TorchAO nvfp4", r["exact"], r)
    except ImportError as e:
        print("SKIP bit-exact TorchAO comparison (TorchAO not installed; not a pass):", e)
    # rotation: orthogonal, matches the existing QuaRot hadamard.py (if importable)
    for K, b in ((512, 512), (2048, 2048), (8192, 8192), (1024, 16), (1024, 128)):
        rot = BlockHadamard(K, b, seed=7)
        x = torch.randn(5, K, dtype=torch.float64)
        rot.ha, rot.hb = rot.ha.double(), (rot.hb.double() if rot.hb is not None else None)
        rot.compute_dtype = torch.float64
        y = rot.apply(x)
        check(f"rotation orthogonal K={K} b={b}", torch.allclose(y.norm(dim=1), x.norm(dim=1)))
    try:
        sys.path.insert(0, "QuaRot")  # optional cross-check against a local QuaRot checkout
        from hadamard import RandomHadamardRotation
        for K in (512, 4096):
            ref = RandomHadamardRotation(K, seed=20260809 + 3)
            mine = BlockHadamard(K, K, seed=20260809 + 3)
            x = torch.randn(4, K)
            check(f"rotation == QuaRot hadamard.py K={K}", torch.allclose(ref.right_rotate(x), mine.apply(x), atol=1e-5))
    except ImportError:
        print("SKIP QuaRot hadamard comparison (not on server)")
    # GPTQ beats RTN on correlated inputs, in the proxy that GPTQ minimises
    torch.manual_seed(0)
    K, N, O = 256, 4096, 64
    mix = torch.randn(K, K) / K ** 0.5 + torch.eye(K)
    X = torch.randn(N, K) @ mix
    X[:, :8] *= 20                                                   # outlier channels
    W = torch.randn(O, K) * 0.05
    rot = BlockHadamard(K, K, seed=1)
    C = X.t() @ X
    Wr = rot.apply(W)
    Hr = transform_hessian(C, rot)
    p_rtn, s_rtn = rtn_weight(Wr)
    p_g, s_g, _ = gptq_nvfp4(Wr, Hr)
    e_rtn, e_g = proxy_error(Wr, p_rtn * s_rtn, Hr), proxy_error(Wr, p_g * s_g, Hr)
    check("gptq < rtn proxy error", e_g < 0.8 * e_rtn, f"rtn {e_rtn:.3g} gptq {e_g:.3g}")
    # every GPTQ value lies on the NVFP4 grid of its own block scale
    deq = p_g * s_g
    re = qdq_blocks(deq, s_g)
    check("gptq output is NVFP4-representable", torch.allclose(re, deq, atol=1e-6, rtol=0))
    # Sim linear == explicit fake-quant math
    lin = torch.nn.Linear(K, O)
    p, s = rtn_weight(rot.apply(lin.weight.detach().float()))
    sim = SimNVFP4Linear(p, s, lin.bias, rotation=BlockHadamard(K, K, seed=1), rot_dtype=torch.float32)
    x = torch.randn(3, 10, K)
    y = sim(x)
    xr = BlockHadamard(K, K, seed=1).apply(x.reshape(-1, K).float())
    ref = (qdq_blocks(xr) @ (p * s).t() + lin.bias.detach()).reshape(3, 10, O)
    # the BF16 GEMM of exact BF16 operands returns a BF16 output (8 significant bits): tolerance 2^-7 relative
    check("sim linear matches fake-quant reference", torch.allclose(y.float(), ref, rtol=2 ** -7, atol=1e-3),
          (y.float() - ref).abs().max().item())
    yf = lin(x)
    check("sim linear close to FP linear", ((y.float() - yf).norm() / yf.norm()).item() < 0.2,
          ((y.float() - yf).norm() / yf.norm()).item())
    sys.path.insert(0, "QuaRot/nvfp4")  # optional
    try:
        from nvfp4_linear import QuaRotNVFP4Linear
    except ImportError:
        QuaRotNVFP4Linear = None
        print("SKIP native NVFP4 comparison (not on server)")
    if torch.cuda.is_available() and QuaRotNVFP4Linear is not None:
        try:
            lin2 = torch.nn.Linear(1024, 512).cuda().to(torch.bfloat16)
            w0 = lin2.weight.detach().clone()
            rotn = BlockHadamard(1024, 1024, seed=11, device="cuda", compute_dtype=torch.bfloat16)
            lin3 = torch.nn.Linear(1024, 512).cuda().to(torch.bfloat16)
            lin3.weight.data.copy_(w0)
            lin3.bias.data.copy_(lin2.bias.data)
            nat = QuaRotNVFP4Linear(lin3, seed=11)
            p, s = rtn_weight(rotn.apply(w0))
            sim = SimNVFP4Linear(p, s, lin2.bias, rotation=rotn, rot_dtype=torch.bfloat16)
            x = torch.randn(4096, 1024, device="cuda", dtype=torch.bfloat16)
            yn, ys = nat(x).float(), sim(x).float()
            rel = ((yn - ys).norm() / yn.norm()).item()
            check("sim vs native NVFP4 linear (rel diff < 1e-2)", rel < 1e-2, rel)
        except Exception as e:                                   # noqa: BLE001
            check("sim vs native NVFP4 linear", False, repr(e))
    print("ALL PASS" if not FAIL else f"{len(FAIL)} FAILED: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
