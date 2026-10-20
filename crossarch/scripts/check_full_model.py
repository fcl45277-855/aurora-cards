"""Check of the full ClimaX model on one real holdout input: the pretrained checkpoint loads strictly, the 38 input
and 5 output variables of the frozen plan are accepted, the upstream loss runs and gradients reach every parameter.

Per-sample standardisation stands in for the training statistics (not yet computed); this checks shapes and
plumbing only, never forecast quality.

    python check_full_model.py            # on the GPU machine (first step after the GPU is switched on)
    python check_full_model.py --cpu      # CPU only with >= 4 GiB free: the 2 GiB CPU-mode container is too small

Never run it in CPU mode next to the training-data download: the container's 2 GiB limit is shared and an
out-of-memory kill could hit the download instead.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_climax_era5 import TASK, frozen_plan, import_climax, write_json  # noqa: E402


def main():
    torch.set_num_threads(2)
    device = 'cpu' if '--cpu' in sys.argv else 'cuda'
    if device == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('GPU not enabled; pass --cpu only where at least 4 GiB of memory is free')
    plan = frozen_plan()
    model_support, lat_weighted_mse = import_climax()
    model = model_support.build_climax()
    model_support.load_climax(model, plan['pretrained']['path'], plan['pretrained']['sha256'])
    holdout = json.loads((TASK / 'holdout_manifest.json').read_text())['inputs']['plumbing'][0]
    with np.load(holdout['path'], allow_pickle=False) as d:
        assert d['variables'].tolist() == plan['variables'] and d['out_variables'].tolist() == plan['out_variables']
        x, y, lat = d['x'], d['y'], d['lat']
    norm = lambda a: (a - a.mean((1, 2), keepdims=True)) / a.std((1, 2), keepdims=True)
    x = torch.tensor(norm(x)[None], device=device); y = torch.tensor(norm(y)[None], device=device)
    lead = torch.tensor([plan['lead_hours'] / 100], device=device)
    model.to(device).train()
    loss_dict, pred = model(x, y, lead, tuple(plan['variables']), plan['out_variables'], [lat_weighted_mse], lat=lat)
    loss = loss_dict[0]['loss']
    loss.backward()
    no_grad = [n for n, p in model.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]
    # Embeddings of the 10 pretrained variables absent from the plan (relative humidity, constants) get no gradient.
    used = set(plan['variables'])
    unexpected = [n for n in no_grad if not any(n.startswith(f'token_embeds.{i}.')
                                                for i, v in enumerate(model.default_vars) if v not in used)]
    report = {'pred_shape': list(pred.shape), 'loss': float(loss.detach()), 'parameters': sum(p.numel() for p in model.parameters()),
              'params_without_grad': len(no_grad), 'unexpected_without_grad': unexpected,
              'device': device, 'unused_pretrained_variables': [v for v in model.default_vars if v not in used]}
    ok = list(pred.shape) == [1, 5, 32, 64] and np.isfinite(report['loss']) and not unexpected
    report['passed'] = bool(ok)
    write_json(ROOT / 'metadata/check_full_model.json', report)
    print(json.dumps(report, indent=2))
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
