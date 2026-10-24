"""Storage of the deployed t7_a16 model as an actual artefact (storage of the deployed model as a file).

Builds native t7_a16 exactly as in Table 8 (TorchAO NVFP4 weights from the simulator's codes and scales, W4A16
kernel, smoothing and low rank in BF16, the remaining backbone layers BF16-resident, everything else as Aurora runs
it), then
  1. lists every parameter and buffer with dtype and bytes (NVFP4 tensors: packed codes, block scales, tensor scale),
  2. saves three artefacts (all parameters and buffers, NVFP4 tensors included) with torch.save and records their sizes:
       as_run         every tensor in the dtype it runs in (FP32 where Aurora keeps FP32),
       packed_linear  weights and biases of the unquantised nn.Linear layers stored in BF16 (the accounting of Table
                      S14f; under autocast these layers compute in BF16 anyway), everything else as run,
       packed_all     every FP32 tensor stored in BF16, including norms, embeddings and buffers,
     BF16-stored tensors are restored to their run dtype after loading (values stay BF16-rounded),
  3. loads each artefact into a freshly built model (every tensor replaced) and scores the test month (2 steps).

  AQ_MODEL=pretrained AQ_PERCEIVER_CHUNK=16384 python scripts/storage_check.py [--split test]
Output: results/_storage_check_t7_a16.json and results/storage_<mode>_t7_a16__<split>_h2/{records.jsonl,summary.json}
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cards import harness as H          # noqa: E402
from cards.recipe import build          # noqa: E402
from scripts.native_t7 import replace_sim_layers       # noqa: E402
from scripts.native_w4a16 import read_jsonl, tensor_bytes  # noqa: E402


def inventory(model: nn.Module) -> dict:
    by_dtype, rows = defaultdict(int), []
    seen = set()
    for mname, mod in model.named_modules():
        for kind, items in (("param", mod.named_parameters(recurse=False)), ("buffer", mod.named_buffers(recurse=False))):
            for tname, t in items:
                if t is None or id(t) in seen:
                    continue
                seen.add(id(t))
                if hasattr(t, "qdata"):                                    # TorchAO NVFP4Tensor
                    parts = {"nvfp4_codes": t.qdata, "e4m3_block_scales": t.scale}
                    if getattr(t, "per_tensor_scale", None) is not None:
                        parts["fp32_tensor_scale"] = t.per_tensor_scale
                    for k, p in parts.items():
                        b = p.numel() * p.element_size()
                        by_dtype[k] += b
                    rows.append((f"{mname}.{tname}", "nvfp4", sum(p.numel() * p.element_size() for p in parts.values())))
                else:
                    b = t.numel() * t.element_size()
                    by_dtype[str(t.dtype).replace("torch.", "")] += b
                    rows.append((f"{mname}.{tname}", str(t.dtype).replace("torch.", ""), b))
    return {"bytes_by_kind": dict(by_dtype), "total_bytes": sum(by_dtype.values()), "n_tensors": len(rows), "tensors": rows}


def tensors(model: nn.Module) -> dict:
    """Every parameter and buffer (also non-persistent ones), keyed '<module>|<name>|param|buffer'."""
    out = {}
    for mname, mod in model.named_modules():
        for tname, t in mod._parameters.items():
            if t is not None:
                out[f"{mname}|{tname}|param"] = t.detach()
        for tname, t in mod._buffers.items():
            if t is not None:
                out[f"{mname}|{tname}|buffer"] = t
    return out


def load_into(model: nn.Module, saved: dict, run_dtype: dict | None = None) -> int:
    """Overwrite every tensor of a freshly built model with the saved one; run_dtype restores BF16-packed tensors."""
    mods = dict(model.named_modules())
    n = 0
    with torch.no_grad():
        for key, t in saved.items():
            mname, tname, kind = key.split("|")
            mod = mods[mname]
            if run_dtype is not None and not hasattr(t, "qdata") and (mname, tname) in run_dtype:
                t = t.to(run_dtype[(mname, tname)])
            if kind == "param":
                setattr(mod, tname, nn.Parameter(t.to("cuda"), requires_grad=False))
            else:
                mod._buffers[tname] = t.to("cuda")
            n += 1
    return n


def build_native(rec):
    from torchao.prototype.mx_formats import NVFP4WeightOnlyConfig
    model, _ = build(rec)
    replace_sim_layers(model, NVFP4WeightOnlyConfig(), "w4a16", weights="from-codes")
    mods = dict(model.named_modules())
    for n in H.layer_set("q0:inc=backbone"):
        if isinstance(mods.get(n), nn.Linear):
            mods[n].to(torch.bfloat16)
    return model


def score(model, name, split, res, rec):
    out = res / f"{name}__{split}_h2"
    out.mkdir(parents=True, exist_ok=True)
    rec_path = out / "records.jsonl"
    rec_path.write_text("")
    truth, cache = H.Truth(), res / "_ref_fp32_h2"
    with rec_path.open("a", encoding="utf-8") as f:
        for d in H.SPLITS[split]:
            preds, lat = H.rollout(model, d, 2, bool(rec.get("surface_fp32", False)))
            for row in H.score(preds, lat, truth, H.load_ref(cache, d, 2)):
                row.update(date=d)
                f.write(json.dumps(row) + "\n")
            del preds
            H.free()
    base = {"fp32": read_jsonl(res / f"fp32__{split}_h2" / "records.jsonl")}
    s = H.summarize(read_jsonl(rec_path), base)
    (out / "summary.json").write_text(json.dumps(s, indent=1), encoding="utf-8")
    return {k: s.get(k) for k in ("qerr_pct", "excess_pct", "excess_pct_ci95", "n_dates")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    a = ap.parse_args()
    res = ROOT / "results"
    art = res / "_artefacts"
    art.mkdir(exist_ok=True)
    rec = json.loads((ROOT / "recipes" / "t7_a16.json").read_text())
    model = build_native(rec)
    report = {"recipe": "t7_a16", "construction": "from-codes, W4A16, remaining backbone layers BF16-resident"}
    report["inventory_as_run"] = inventory(model)
    run_dtype = {}
    for mname, mod in model.named_modules():
        for tname, t in list(mod.named_parameters(recurse=False)) + list(mod.named_buffers(recurse=False)):
            if t is not None and not hasattr(t, "qdata"):
                run_dtype[(mname, tname)] = t.dtype
    torch.cuda.synchronize()
    p_run = art / "t7_a16_as_run.pt"
    torch.save(tensors(model), p_run)
    report["file_bytes_as_run"] = p_run.stat().st_size
    paths = {"as_run": p_run}
    with torch.no_grad():                    # packed_linear: unquantised Linear weights and biases in BF16
        for mname, mod in model.named_modules():
            if isinstance(mod, nn.Linear):
                for tname, t in mod.named_parameters(recurse=False):
                    if not hasattr(t, "qdata") and t.dtype == torch.float32:
                        t.data = t.data.to(torch.bfloat16)
    report["inventory_packed_linear"] = {k: v for k, v in inventory(model).items() if k != "tensors"}
    paths["packed_linear"] = art / "t7_a16_packed_linear.pt"
    torch.save(tensors(model), paths["packed_linear"])
    report["file_bytes_packed_linear"] = paths["packed_linear"].stat().st_size
    with torch.no_grad():                    # packed_all: every remaining FP32 tensor in BF16
        for mname, mod in model.named_modules():
            for tname, t in list(mod.named_parameters(recurse=False)) + list(mod.named_buffers(recurse=False)):
                if t is not None and not hasattr(t, "qdata") and t.is_floating_point() and t.dtype == torch.float32:
                    t.data = t.data.to(torch.bfloat16)
    report["inventory_packed_all"] = {k: v for k, v in inventory(model).items() if k != "tensors"}
    paths["packed_all"] = art / "t7_a16_packed_all.pt"
    torch.save(tensors(model), paths["packed_all"])
    report["file_bytes_packed_all"] = paths["packed_all"].stat().st_size
    bf16_ref = sum(dims["in_features"] * dims["out_features"] * 2 for dims in
                   json.loads(H.MANIFEST.read_text(encoding="utf-8"))["dimensions"].values())
    report["bf16_linear_weight_bytes_reference"] = bf16_ref
    del model
    H.free()
    for mode, path in paths.items():
        t0 = time.time()
        m = build_native(rec)                                     # skeleton; every tensor is replaced below
        saved = torch.load(path, weights_only=False, map_location="cpu")
        report[f"reload_{mode}_tensors"] = load_into(m, saved, run_dtype if mode != "as_run" else None)
        del saved
        report[f"reload_{mode}"] = score(m, f"storage_{mode}_t7_a16", a.split, res, rec)
        report[f"reload_{mode}"]["seconds"] = time.time() - t0
        del m
        H.free()
        print(json.dumps({mode: report[f"reload_{mode}"]}), flush=True)
    report["inventory_as_run"]["tensors"] = report["inventory_as_run"]["tensors"][:0]   # keep the file small
    (res / "_storage_check_t7_a16.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("file_bytes_as_run", "file_bytes_packed_linear", "file_bytes_packed_all",
                                             "bf16_linear_weight_bytes_reference")}))


if __name__ == "__main__":
    main()
