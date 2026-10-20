"""Fine-tune the public ClimaX 5.625-degree CMIP6 checkpoint on WeatherBench2 ERA5 for a 6 h forecast.

Implements exactly the optimisation frozen in finetune_era5_20261020/task_plan.json, written before any GPU
training, and the forecast gate stated there:
  train 2010-2015 (8763 pairs), validate 2016 (1463 pairs); AdamW lr 1e-4, weight decay 0.05; micro-batch 2,
  4 accumulation steps; 1 warm-up epoch then cosine to 1e-6; gradient clip 1.0; FP32 (TF32 off); full fine-tuning;
  20 epochs at most, early stopping after epoch 5 with patience 4; the checkpoint with the minimum 2016
  latitude-weighted standardised MSE is selected; it must beat 6 h persistence on the aggregate and on z500 and t850.

The loss is ClimaX's lat_weighted_mse over the five output variables in standardised units, as upstream
global_forecast. The run is resumable at epoch boundaries and deterministic for a fixed device.

    python train_climax_era5.py --check     # CPU: frozen plan, data manifest, sources and weights; no training
    python train_climax_era5.py --smoke     # CPU: tiny model on synthetic states, exercises the whole loop
    python train_climax_era5.py             # GPU: the real fine-tuning (resumes from RUN/resume.pt if present)

Outputs in RUN: best.ckpt (state_dict with the upstream "net." prefix), resume.pt, history.json, provenance.json.
make_climax_config.py turns provenance.json into the config that run_climax.py's asset gate accepts.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from crossarch_paths import ROOT
TASK = ROOT / 'finetune_era5_20261020'
HERE = Path(__file__).resolve().parent
RUN = Path(os.environ.get('CLIMAX_FT_RUN', ROOT / 'climax_finetune_run'))

# The optimisation this script implements; --check refuses to run if the frozen plan says anything else.
FROZEN_OPTIMIZATION = {
    'seed': 20261020, 'optimizer': 'AdamW', 'lr': 0.0001, 'weight_decay': 0.05, 'batch_size': 2, 'accumulate': 4,
    'max_epochs': 20, 'min_epochs': 5, 'early_stopping_patience': 4, 'warmup_epochs': 1,
    'schedule': 'cosine to 1e-6', 'gradient_clip': 1.0, 'precision': 'FP32', 'full_finetuning': True,
    'checkpoint_selection': 'Minimum 2016 latitude-weighted standardized MSE only'}
WARMUP_START_LR = 1e-8                       # upstream ClimaX default warmup_start_lr
FINAL_LR = 1e-6
GATE_VARIABLES = ['geopotential_500', 'temperature_850']
EVAL_BATCH = 16
UPSTREAM_METRICS_SHA256 = 'a138fc1ec68db5b8f515f78ac5c2f2e83831471d0885c3fbdd652795e37a4d59'


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.partial')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def import_climax():
    """Use the preparation tree's isolated compatibility copy of ClimaX, verified against its recorded hashes."""
    records = json.loads((ROOT / 'metadata/compatibility_changes.json').read_text())
    for r in records:
        if 'copy' in r and sha(r['copy']) != r['copy_sha256']:
            raise RuntimeError('Compatibility copy changed since it was recorded: ' + r['copy'])
    sys.path.insert(0, str(ROOT / 'sources/compat'))
    import model_support                                            # noqa: E402  (read-only helpers)
    # The upstream loss, verbatim: src/climax/utils/metrics.py at ClimaX commit 6d5d354 (git blob bb1bb3b8).
    # The preparation tree's source capture does not include it, so a byte-identical copy sits next to this script.
    import importlib.util
    path = HERE / 'climax_upstream_metrics.py'
    if sha(path) != UPSTREAM_METRICS_SHA256:
        raise RuntimeError('climax_upstream_metrics.py differs from the pinned upstream file')
    spec = importlib.util.spec_from_file_location('climax_upstream_metrics', path)
    metrics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metrics)
    return model_support, metrics.lat_weighted_mse


# ----------------------------------------------------------------------------------------------------------------
# data

class PairSet:
    """Consecutive 6 h state pairs (t, t+1) with both states inside [start, stop), standardised on the fly."""

    def __init__(self, states, start, stop, mean, std, out_index):
        self.states, self.mean, self.std, self.out = states, mean, std, out_index
        self.t = np.arange(start, stop - 1)

    def __len__(self):
        return len(self.t)

    def batch(self, rows, device):
        t = np.sort(self.t[rows])                                   # sorted reads from the memmap; order is irrelevant
        x = (np.asarray(self.states[t], dtype=np.float32) - self.mean) / self.std
        y = (np.asarray(self.states[t + 1], dtype=np.float32) - self.mean) / self.std
        x = torch.from_numpy(x).to(device)
        y = torch.from_numpy(y[:, self.out]).to(device)
        return x, y, x[:, self.out]                                 # input, target, persistence


def load_statistics(variables):
    with np.load(TASK / 'normalize_mean.npz', allow_pickle=False) as m, \
            np.load(TASK / 'normalize_std.npz', allow_pickle=False) as s:
        mean = np.array([float(m[v]) for v in variables], dtype=np.float32)[:, None, None]
        std = np.array([float(s[v]) for v in variables], dtype=np.float32)[:, None, None]
    if not (np.isfinite(mean).all() and np.isfinite(std).all() and (std > 0).all()):
        raise ValueError('Invalid training statistics')
    return mean, std


# ----------------------------------------------------------------------------------------------------------------
# training

def learning_rate(step, steps_per_epoch, opt):
    warm = opt['warmup_epochs'] * steps_per_epoch
    if step < warm:
        return WARMUP_START_LR + (opt['lr'] - WARMUP_START_LR) * step / warm
    total = opt['max_epochs'] * steps_per_epoch
    progress = (step - warm) / max(1, total - warm)
    return FINAL_LR + 0.5 * (opt['lr'] - FINAL_LR) * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model, data, lead, variables, outputs, lat, device):
    """Latitude-weighted standardised MSE per output variable over all pairs, model and persistence."""
    w = torch.cos(torch.deg2rad(torch.as_tensor(lat, dtype=torch.float64, device=device)))
    w = (w / w.mean())[None, None, :, None]
    model_sum = torch.zeros(len(outputs), dtype=torch.float64, device=device)
    pers_sum = torch.zeros_like(model_sum)
    for begin in range(0, len(data), EVAL_BATCH):
        rows = np.arange(begin, min(begin + EVAL_BATCH, len(data)))
        x, y, p = data.batch(rows, device)
        pred = model(x, None, lead.expand(len(rows)), variables, outputs, None, None)[1]
        model_sum += ((pred.double() - y.double()) ** 2 * w).mean((2, 3)).sum(0)
        pers_sum += ((p.double() - y.double()) ** 2 * w).mean((2, 3)).sum(0)
    m = (model_sum / len(data)).tolist()
    p = (pers_sum / len(data)).tolist()
    return {'model': dict(zip(outputs, m)), 'persistence': dict(zip(outputs, p)),
            'model_aggregate': float(np.mean(m)), 'persistence_aggregate': float(np.mean(p))}


def forecast_gate(metrics, gate_variables):
    checks = {'aggregate': metrics['model_aggregate'] < metrics['persistence_aggregate']}
    for v in gate_variables:
        checks[v] = metrics['model'][v] < metrics['persistence'][v]
    return {'passed': all(checks.values()), 'checks': checks}


def save_net(model, path, extra):
    state = {'state_dict': {'net.' + k: v.detach().cpu() for k, v in model.state_dict().items()}, **extra}
    tmp = Path(str(path) + '.partial')
    torch.save(state, tmp)
    tmp.replace(path)


def train(*, model, lat_weighted_mse, train_set, val_set, variables, outputs, lat, opt, device, run, gate_variables,
          lead_hours, stop_after_epochs=None, log=print):
    run.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(opt['seed'])
    variables, outputs = tuple(variables), list(outputs)
    lead = torch.tensor([lead_hours / 100], dtype=torch.float32, device=device)   # upstream hours/100 embedding
    lat_np = np.asarray(lat, dtype=np.float64)
    params = [p for p in model.parameters()]
    for p in params:
        p.requires_grad_(True)                                      # full fine-tuning
    optimizer = torch.optim.AdamW(params, lr=opt['lr'], weight_decay=opt['weight_decay'])
    group = opt['batch_size'] * opt['accumulate']
    steps_per_epoch = len(train_set) // group                       # the remainder of each shuffled epoch is skipped
    state = {'epoch': 0, 'step': 0, 'best': None, 'best_epoch': None, 'since_best': 0, 'history': []}
    resume = run / 'resume.pt'
    if resume.exists():
        saved = torch.load(resume, map_location='cpu', weights_only=False)
        model.load_state_dict(saved['model'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        state = saved['state']
        log(f'resumed at epoch {state["epoch"]}')
    model.to(device)
    for g in optimizer.state.values():
        for k, v in list(g.items()):
            if torch.is_tensor(v):
                g[k] = v.to(device)

    while state['epoch'] < opt['max_epochs']:
        epoch = state['epoch']
        model.train()
        order = np.random.default_rng(opt['seed'] + epoch).permutation(len(train_set))
        running, seen = 0.0, 0
        for s in range(steps_per_epoch):
            lr = learning_rate(state['step'], steps_per_epoch, opt)
            for g in optimizer.param_groups:
                g['lr'] = lr
            optimizer.zero_grad(set_to_none=True)
            for a in range(opt['accumulate']):
                rows = order[(s * opt['accumulate'] + a) * opt['batch_size']:][:opt['batch_size']]
                x, y, _ = train_set.batch(rows, device)
                loss_dict, _ = model(x, y, lead.expand(len(rows)), variables, outputs, [lat_weighted_mse], lat=lat_np)
                loss = loss_dict[0]['loss']
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'non-finite loss at epoch {epoch} step {s}')
                (loss / opt['accumulate']).backward()
                running += loss.item() * len(rows)
                seen += len(rows)
            torch.nn.utils.clip_grad_norm_(params, opt['gradient_clip'])
            optimizer.step()
            state['step'] += 1
        model.eval()
        metrics = evaluate(model, val_set, lead, variables, outputs, lat_np, device)
        row = {'epoch': epoch, 'train_loss': running / max(seen, 1), 'lr_end': lr, **metrics}
        improved = state['best'] is None or metrics['model_aggregate'] < state['best']
        if improved:
            state.update(best=metrics['model_aggregate'], best_epoch=epoch, since_best=0)
            save_net(model, run / 'best.ckpt', {'epoch': epoch, 'validation': metrics})
        else:
            state['since_best'] += 1
        state['history'].append(row)
        state['epoch'] = epoch + 1
        log(f'epoch {epoch}: train {row["train_loss"]:.5f} val {metrics["model_aggregate"]:.5f} '
            f'(persistence {metrics["persistence_aggregate"]:.5f}){" *" if improved else ""}')
        stop = state['epoch'] >= opt['min_epochs'] and state['since_best'] >= opt['early_stopping_patience']
        state['stopped_early'] = bool(stop)
        tmp = Path(str(resume) + '.partial')
        torch.save({'model': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    'optimizer': optimizer.state_dict(), 'state': state}, tmp)
        tmp.replace(resume)
        write_json(run / 'history.json', state['history'])
        if stop or (stop_after_epochs is not None and state['epoch'] >= stop_after_epochs):
            break
    best = state['history'][state['best_epoch']]
    return {'steps_per_epoch': steps_per_epoch, 'epochs_run': state['epoch'], 'best_epoch': state['best_epoch'],
            'stopped_early': state.get('stopped_early', False), 'validation_at_best': best,
            'forecast_gate': forecast_gate(best, gate_variables)}


# ----------------------------------------------------------------------------------------------------------------
# modes

def frozen_plan():
    plan = json.loads((TASK / 'task_plan.json').read_text())
    if plan['optimization'] != FROZEN_OPTIMIZATION:
        raise RuntimeError('task_plan.json optimisation differs from what this script implements')
    if plan['lead_hours'] != 6 or plan['lead_embedding_divisor'] != 100:
        raise RuntimeError('Unexpected lead time in the frozen plan')
    return plan


def check(require_data=True):
    problems, info = [], {}
    plan = frozen_plan()
    info['task_plan_sha256'] = sha(TASK / 'task_plan.json')
    if sha(plan['pretrained']['path']) != plan['pretrained']['sha256']:
        problems.append('pretrained checkpoint SHA256 mismatch')
    manifest = TASK / 'training_data_manifest.json'
    if not manifest.exists():
        problems.append('training data not complete yet (no training_data_manifest.json)')
    elif require_data:
        m = json.loads(manifest.read_text())
        if not m.get('complete') or m['task_plan_sha256'] != info['task_plan_sha256']:
            problems.append('training data manifest incomplete or for another plan')
        for key in ['states', 'times', 'mean', 'std']:
            if sha(m[key]['path']) != m[key]['sha256']:
                problems.append(key + ': SHA256 mismatch')
        info['training_data_manifest_sha256'] = sha(manifest)
    import_climax()
    info['cuda'] = torch.cuda.is_available()
    if RUN.exists() and (RUN / 'provenance.json').exists():
        problems.append('a finished run already exists in ' + str(RUN))
    return problems, info


def real():
    problems, info = check()
    if problems:
        raise RuntimeError('; '.join(problems))
    if not torch.cuda.is_available():
        raise RuntimeError('GPU is not enabled; fine-tuning not started')
    plan = frozen_plan()
    model_support, lat_weighted_mse = import_climax()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    variables, outputs = plan['variables'], plan['out_variables']
    states = np.load(TASK / 'states.npy', mmap_mode='r')
    times = np.load(TASK / 'times.npy')
    assert states.shape == tuple(plan['state_shape']) and len(times) == len(states)
    cut = plan['split_index']
    assert str(times[cut])[:13] == '2016-01-01T00'
    mean, std = load_statistics(variables)
    out_index = [variables.index(v) for v in outputs]
    train_set = PairSet(states, 0, cut, mean, std, out_index)
    val_set = PairSet(states, cut, len(states), mean, std, out_index)
    assert len(train_set) == plan['training']['pairs'] and len(val_set) == plan['validation']['pairs']
    lat = np.load(TASK / 'latitude.npy')
    model = model_support.build_climax()
    model_support.load_climax(model, plan['pretrained']['path'], plan['pretrained']['sha256'])
    RUN.mkdir(parents=True, exist_ok=True)
    logf = (RUN / 'train.log').open('a')

    def log(msg):
        line = dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds') + ' ' + msg
        print(line, flush=True)
        logf.write(line + '\n')
        logf.flush()
    log(f'start on {torch.cuda.get_device_name()} with {len(train_set)} train / {len(val_set)} validation pairs')
    summary = train(model=model, lat_weighted_mse=lat_weighted_mse, train_set=train_set, val_set=val_set,
                    variables=variables, outputs=outputs, lat=lat, opt=plan['optimization'], device='cuda', run=RUN,
                    gate_variables=GATE_VARIABLES, lead_hours=plan['lead_hours'], log=log)
    import timm
    provenance = {
        'task': 'era5_finetuned', 'task_id': plan['task_id'], 'scope': plan['scope'],
        'utc_finished': dt.datetime.now(dt.timezone.utc).isoformat(),
        'checkpoint': {'path': str(RUN / 'best.ckpt'), 'sha256': sha(RUN / 'best.ckpt')},
        'pretrained': plan['pretrained'], 'task_plan_sha256': info['task_plan_sha256'],
        'training_data_manifest_sha256': info['training_data_manifest_sha256'],
        'script_sha256': sha(__file__), 'train_log_sha256': sha(RUN / 'train.log'),
        'history_sha256': sha(RUN / 'history.json'),
        'device': torch.cuda.get_device_name(), 'torch': torch.__version__, 'timm': timm.__version__,
        'variables': variables, 'out_variables': outputs, 'lead_hours': plan['lead_hours'],
        'lead_embedding_divisor': 100, 'dropout': 'drop_rate = drop_path = 0 (model_support.build_climax)',
        **summary}
    write_json(RUN / 'provenance.json', provenance)
    log('forecast gate ' + ('PASSED' if summary['forecast_gate']['passed'] else 'FAILED: stop before quantisation'))
    return 0 if summary['forecast_gate']['passed'] else 3


def smoke():
    """Tiny ClimaX on synthetic advected fields: the loop runs, learns, checkpoints reload, and resume is exact."""
    model_support, lat_weighted_mse = import_climax()
    torch.set_num_threads(2)
    variables = ['2m_temperature', '10m_u_component_of_wind', '10m_v_component_of_wind']
    rng = np.random.default_rng(0)
    T, H, W = 400, 8, 16
    base = rng.normal(size=(len(variables), H, W)).astype(np.float32)
    states = np.stack([np.roll(base, t, axis=-1) + 0.05 * rng.normal(size=base.shape) for t in range(T)])
    states = (states * np.array([10, 3, 3], dtype=np.float32)[:, None, None]
              + np.array([280, 0, 0], dtype=np.float32)[:, None, None]).astype(np.float32)
    mean = states[:300].mean((0, 2, 3))[:, None, None]
    std = states[:300].std((0, 2, 3))[:, None, None]
    lat = np.linspace(-78.75, 78.75, H)
    opt = dict(FROZEN_OPTIMIZATION, lr=1e-3, max_epochs=4, min_epochs=2, early_stopping_patience=2)
    out_index = [0, 1, 2]
    gate = variables[:2]

    def fresh(seed=1):
        torch.manual_seed(seed)
        return model_support.build_climax(tiny=True)
    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        tr = PairSet(states, 0, 300, mean, std, out_index)
        va = PairSet(states, 300, T, mean, std, out_index)
        common = dict(lat_weighted_mse=lat_weighted_mse, train_set=tr, val_set=va, variables=variables,
                      outputs=variables, lat=lat, opt=opt, device='cpu', gate_variables=gate, lead_hours=6,
                      log=lambda m: None)
        straight = train(model=fresh(), run=tmp / 'a', **common)
        m_a = fresh()
        state = torch.load(tmp / 'a' / 'best.ckpt', weights_only=True)['state_dict']
        m_a.load_state_dict(model_support.canonical_climax_state(state), strict=True)
        m_a.eval()
        again = evaluate(m_a, va, torch.tensor([0.06]), tuple(variables), variables, lat, 'cpu')
        partial = train(model=fresh(), run=tmp / 'b', stop_after_epochs=2, **common)
        resumed = train(model=fresh(seed=99), run=tmp / 'b', **common)    # weights come from resume.pt
        hist = json.loads((tmp / 'a' / 'history.json').read_text())
        results = {
            'learns': all(b['train_loss'] < a['train_loss'] for a, b in zip(hist, hist[1:])),
            'beats_persistence': straight['forecast_gate']['passed'],
            'best_reload_exact': abs(again['model_aggregate'] - straight['validation_at_best']['model_aggregate']) < 1e-9,
            'resume_exact': resumed['validation_at_best'] == straight['validation_at_best']
                            and resumed['epochs_run'] == straight['epochs_run'],
            'partial_stopped_at_2': partial['epochs_run'] == 2,
            'lr_schedule': [learning_rate(s, 10, FROZEN_OPTIMIZATION) for s in (0, 5, 10, 200)],
            'train_loss_by_epoch': [h['train_loss'] for h in hist],
            'val_vs_persistence': [straight['validation_at_best']['model_aggregate'],
                                   straight['validation_at_best']['persistence_aggregate']]}
    lr = results['lr_schedule']
    results['lr_ok'] = abs(lr[0] - WARMUP_START_LR) < 1e-15 and abs(lr[2] - 1e-4) < 1e-12 and abs(lr[3] - FINAL_LR) < 1e-12
    ok = all(results[k] for k in ['learns', 'beats_persistence', 'best_reload_exact', 'resume_exact',
                                  'partial_stopped_at_2', 'lr_ok'])
    print(json.dumps(results, indent=2))
    print('SMOKE PASS' if ok else 'SMOKE FAIL')
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    if a.smoke:
        return smoke()
    if a.check:
        problems, info = check()
        print(json.dumps({'ready': not problems, 'problems': problems, **info, 'run_dir': str(RUN)}, indent=2))
        return 0 if not problems else 2
    return real()


if __name__ == '__main__':
    raise SystemExit(main())
