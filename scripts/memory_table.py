"""Weight-storage accounting (bytes) for the paper's main configurations. Simulated quantization: these are the sizes
a packed deployment would store, not measured GPU memory (peak memory is dominated by activations).

  NVFP4 weight: 4-bit E2M1 + one E4M3 scale per 16 elements = 0.5625 B/param (+ FP32 per-tensor scale)
  FP8 baselines: MXFP8 = E4M3 + one E8M0 scale per 32 elements = 1.03125 B/param; per-channel FP8 = 1 B/param + one FP32
  scale per output channel
  low-rank branch (rank r): r * (in + out) BF16 params; smoothing vector: in FP32; everything else BF16 (2 B/param)

  python scripts/memory_table.py
"""
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cards import harness as H          # noqa: E402

CONFIGS = ["bf16", "q0_sim", "c08_weight_only_gptq", "g02_svdq_a08_r32", "c08_t7set", "t7_g02_nodec_fold", "t7_mx",
           "t7mx_all314_noenc", "t7_a16", "u8_w8a16_mx", "u8_w8a8_mx", "u8_w8a16_ch", "u8_w8a8_ch", "u8_w8a16_mx_prot", "u8_w8a8_mx_prot", "t7a16_all314_noenc"]


def main():
    configs = sys.argv[1:] or CONFIGS
    sd = torch.load(H.CKPT, map_location="cpu", mmap=True, weights_only=False)
    sd = sd.get("state_dict", sd)
    total = sum(v.numel() for v in sd.values() if hasattr(v, "numel"))
    shapes = {}
    for n in H.manifest_layers():
        m = [k for k in sd if k.endswith(n + ".weight")]
        shapes[n] = tuple(sd[m[0]].shape) if m else None
    print(f"total params {total / 1e6:.1f}M; BF16 storage {2 * total / 2 ** 20:.0f} MiB")
    print(f"{'config':22s} {'quant layers':>12s} {'quant params':>13s} {'low-rank':>9s} {'weights MiB':>12s} {'vs BF16':>8s} {'avg bits':>9s}")
    for c in configs:
        r = json.loads((ROOT / "recipes" / f"{c}.json").read_text())
        if r.get("kind") == "bf16":
            names = []
        else:
            names = H.layer_set(r["layers"])
        rank = (r.get("low_rank") or {}).get("rank") or 0
        q = sum(shapes[n][0] * shapes[n][1] for n in names)
        lr = sum(rank * (shapes[n][0] + shapes[n][1]) for n in names)
        sm = sum(shapes[n][1] for n in names) if r.get("smooth") else 0
        method = r.get("weight", {}).get("method", "rtn")
        if method == "mxfp8":
            qbytes = 1.03125 * q
        elif method == "fp8_channel":
            qbytes = 1.0 * q + 4 * sum(shapes[n][0] for n in names)
        else:
            qbytes = 0.5625 * q + 4 * len(names)
        nbytes = qbytes + 2 * lr + 4 * sm + 2 * (total - q)
        print(f"{c:22s} {len(names):12d} {q / 1e6:10.1f}M {lr / 1e6:7.1f}M {nbytes / 2 ** 20:12.0f} {nbytes / (2 * total):8.2f} {8 * nbytes / total:9.2f}")


if __name__ == "__main__":
    main()
