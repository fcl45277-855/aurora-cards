# Tutorial: three typical use cases

1. Define a new quantisation recipe for Aurora and evaluate it.
2. Apply the paper's deployable layer to the linear layers of your own PyTorch model.
3. Run the switching-dominance diagnostic on an attention module of your own model.

Use cases 2 and 3 need no data, checkpoint or GPU. Every code block below is run by `tests/tutorial_check.py`.

---

## 1. A new recipe for Aurora

**Goal:** test whether the deployable recipe still works when the encoder resampler is quantised as well, with 4-bit activations everywhere.

**Step 1: choose the layers.** Layer sets are specified by name (`docs/USER_GUIDE.md`, Sect. 2.3). Check the selection without loading the model:

```python
from cards import harness as H
base = "q0:exc=decoder.level_decoder,decoder.atmos_levels_embed,ln_modulation,time_mlp,_embed"   # t7: 138 layers
print(len(H.layer_set(base)))
enc = [n for n in H.layer_set("all314:inc=encoder.level_agg") if n not in H.layer_set(base)]
print(len(enc), enc)    # the 5 encoder-resampler layers that t7 does not quantise
```

**Step 2: write the recipe.** Save it as `recipes/my_t7_plus_enc_a4.json`:

```json
{"name": "my_t7_plus_enc_a4",
 "layers": "my_layers.txt",
 "smooth": {"alpha": 0.8}, "low_rank": {"rank": 32}, "rotation": {"block": null},
 "weight": {"method": "gptq", "source": "fp32"}, "act_scale_search": [1.0, 1.5], "act": "nvfp4"}
```

A layer set can also be a file with one layer name per line. Write `my_layers.txt` from Python:

```python
from pathlib import Path
Path("my_layers.txt").write_text("\n".join(H.layer_set(base) + enc) + "\n")
```

**Step 3: run it.** It needs the checkpoint, the `calib` days for GPTQ and smoothing, and the days of the split (`docs/DATA.md`):

```bash
export AQ_CKPT_DIR=... AQ_ERA5_DIR=... AQ_PERCEIVER_CHUNK=16384
python scripts/run_card.py --recipe recipes/fp32.json --split dev        # once per split: FP32 reference
python scripts/run_card.py --recipe recipes/my_t7_plus_enc_a4.json --split dev
python scripts/block_ci.py --split dev my_t7_plus_enc_a4 t7_g02_nodec_fold --full
```

**What to expect.** `results/my_t7_plus_enc_a4__dev_h2/summary.json` reports `qerr_pct` and `excess_pct`. The paper's closest card, `t7_plus_enc` (t7 plus the encoder resampler), has a test qerr of 50.1 % against 17.3 % for t7. Expect a large increase here too, because the encoder resampler is one of the two 4-bit hot spots.

---

## 2. The deployable layer in your own model

`cards.deployable` implements Eqs. (3)–(10) and Algorithm 1 of the paper for any `torch.nn.Linear`. It needs the input statistics of each layer over a calibration set.

```python
import torch
from torch import nn
from cards.deployable import accumulate_stats, build_deployable_linear

torch.manual_seed(0)
model = nn.Sequential(nn.Linear(256, 512), nn.GELU(), nn.Linear(512, 256))
calib = [torch.randn(1024, 256) * torch.cat([torch.full((8,), 20.0), torch.ones(248)]) for _ in range(4)]

# 1. collect per-layer input statistics with forward hooks
stats, hooks = {}, []
for name, m in model.named_modules():
    if isinstance(m, nn.Linear):
        hooks.append(m.register_forward_pre_hook(
            lambda mod, inp, name=name: stats.__setitem__(name, accumulate_stats(inp[0], stats.get(name)))))
with torch.no_grad():
    for x in calib:
        model(x)
for h in hooks:
    h.remove()

# 2. replace every Linear by the deployable layer (NVFP4 weights by GPTQ, smoothing, rank-32 branch, MXFP8 activations)
test = calib[0][:256]
with torch.no_grad():
    y_ref = model(test)
for name, m in list(model.named_modules()):
    if isinstance(m, nn.Linear):
        q = build_deployable_linear(m.weight, m.bias, stats[name], alpha=0.8, rank=32, act="mxfp8")
        setattr(model, name, q)
        print(name, "GPTQ objective", round(q.info["proxy_gptq"], 1), "vs RTN", round(q.info["proxy_rtn"], 1))
with torch.no_grad():
    y_q = model(test).float()
print("relative output error", float((y_q - y_ref).norm() / y_ref.norm()))
```

**Options.** `act="nvfp4"` gives the t7 variant with 4-bit activations, `act="bf16"` is weight-only, `method="rtn"` replaces GPTQ, `alpha=None` turns smoothing off and `rank=0` removes the low-rank branch. The returned module is a simulated quantiser (`cards.simlinear.SimNVFP4Linear`) whose numerics match TorchAO's NVFP4 bit for bit. For real 4-bit kernels on Blackwell GPUs see `scripts/native_t7.py`.

**What to expect.** The relative output error is a few per cent. For a whole network, judge the result on the task metric, not the layer error: the paper's central finding is that the layer error does not predict which layers matter (use case 3).

---

## 3. The switching diagnostic on your own attention module

The diagnostic asks how much of the output error of an attention module, under the key error that quantisation causes, is carried by cells whose argmax key switches (`err_in_flip`). Modules where this share is high are fragile (paper Sect. 4.2). It needs the queries, keys and values of one forward pass, split into heads, and the key perturbation.

```python
import torch
from cards.nvfp4 import qdq_blocks
from cards.switching import switching_diagnostic

torch.manual_seed(0)
d_model, heads = 256, 8
d = d_model // heads
to_q, to_k, to_v = (torch.nn.Linear(d_model, d_model) for _ in range(3))
latents = torch.randn(64, 4, d_model)        # 4 learned queries per sample
context = torch.randn(64, 13, d_model) * 3   # 13 keys per sample, e.g. pressure levels

def heads_(t):                                 # [b, L, d_model] -> [b, heads, L, d]
    return t.reshape(t.shape[0], t.shape[1], heads, d).transpose(1, 2)

with torch.no_grad():
    q, k, v = heads_(to_q(latents)), heads_(to_k(context)), heads_(to_v(context))
    k_quant = heads_(to_k(qdq_blocks(context)))          # keys computed from NVFP4-quantised inputs
r = switching_diagnostic(q, k, k_quant - k, v)
print({key: round(val, 3) for key, val in r.items()})
```

**Reading the output.** `flip_rate` is the share of attention cells whose argmax key changes. `err_in_flip` is the share of the output error in those cells. `A` is the common-mode amplification of the logit differences, and `G` is the exact error divided by its linearised prediction. In Aurora the two resamplers give `err_in_flip ≥ 0.94`, and ranking all attention modules by `err_in_flip` matched their measured sensitivity on three models. The local activation error did not.

To use the diagnostic on a real model, capture `q`, `k` and `v` with a forward hook on the attention module. Get `k_quant` by running the key projection under the quantiser you plan to deploy. `scripts/replay_attn.py` does exactly this for Aurora's resamplers.
