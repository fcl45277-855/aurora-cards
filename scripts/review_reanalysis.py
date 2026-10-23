"""Re-analyses requested by the pre-submission review of v9 (2026-10-23), from existing records only (no GPU).

  1. Headline intervals: date bootstrap with blocks of L = 1 (i.i.d.), 3, 5, 7 consecutive dates, non-circular and
     circular, for the 30-date test month (t7_mx, t7, c08); 2000 resamples, percentile intervals, seed 0.
  2. Short sets (8 July dates, 4 March 2023 dates): per-date qerr / excess, mean and range instead of intervals.
  3. Twenty-step rollouts: per-initial-date t7_mx ERA5 excess at 6, 72 and 120 h (test month and July).
  4. Exact number of distinct resamples of the non-circular 3-day block bootstrap for n = 4 and n = 8 dates.

    python scripts/review_reanalysis.py [--results <dir>] > review_reanalysis.txt

The results directory holds the per-card records.jsonl of the full result archive (docs/DATA.md); it defaults to
$AQ_RESULTS, else <repo>/results.
"""
import argparse
import itertools
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    t = {}
    for line in (path / 'records.jsonl').read_text().splitlines():
        r = json.loads(line)
        t.setdefault((r['date'], r['step'], r['var']), r)
    return t


def find(results, name):
    for base in (results / 'results', results):
        if (base / name / 'records.jsonl').exists():
            return base / name
    raise FileNotFoundError(name)


def headline(card, fp, dates, vs):
    ks = [(d, s, v) for d in dates for (s, v) in vs]
    f = np.array([fp[k]['rmse_truth'] for k in ks])
    return (100 * float(np.mean(np.array([card[k]['rmse_ref'] for k in ks]) / f)),
            100 * float(np.mean(np.array([card[k]['rmse_truth'] for k in ks]) / f - 1)))


def resample(dates, rng, L, circular):
    n = len(dates)
    out = []
    while len(out) < n:
        s = int(rng.integers(0, n if circular else n - L + 1))
        out.extend(dates[(s + i) % n] for i in range(L))
    return out[:n]


def intervals(card, fp, dates, vs, L, circular, n=2000):
    rng = np.random.default_rng(0)
    b = np.array([headline(card, fp, resample(dates, rng, L, circular), vs) for _ in range(n)])
    return np.percentile(b[:, 0], [2.5, 97.5]), np.percentile(b[:, 1], [2.5, 97.5])


def pair(results, name, split, steps=2):
    card = load(find(results, f'{name}__{split}_h{steps}'))
    fp = load(find(results, f'fp32__{split}_h{steps}'))
    keys = [k for k in card if k in fp]
    return card, fp, sorted({k[0] for k in keys}), sorted({(k[1], k[2]) for k in keys})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default=os.environ.get('AQ_RESULTS', str(ROOT / 'results')))
    results = Path(ap.parse_args().results)

    print('== 1. Test-month intervals by block length (qerr %, ERA5 excess %)')
    for name in ['t7_mx', 't7_g02_nodec_fold', 'c08_weight_only_gptq']:
        card, fp, dates, vs = pair(results, name, 'test')
        q, e = headline(card, fp, dates, vs)
        print(f'{name}: n={len(dates)} point qerr {q:.2f} excess {e:+.2f}')
        for L in (1, 3, 5, 7):
            for circ in (False, True):
                if L == 1 and circ:
                    continue
                qi, ei = intervals(card, fp, dates, vs, L, circ)
                print(f'   L={L} {"circular" if circ else "non-circular"}: qerr [{qi[0]:.2f}, {qi[1]:.2f}]  '
                      f'excess [{ei[0]:+.2f}, {ei[1]:+.2f}]')

    print('\n== 2. Short sets: per-date values (qerr %, ERA5 excess %)')
    for split, names in (('summer', ['t7_mx', 'c08_weight_only_gptq', 'q0_sim']),
                         ('y2023', ['t7_mx', 'c08_weight_only_gptq', 'q0_sim'])):
        for name in names:
            try:
                card, fp, dates, vs = pair(results, name, split)
            except FileNotFoundError as err:
                print(f'{split} {name}: missing {err}')
                continue
            per = [headline(card, fp, [d], vs) for d in dates]
            q, e = headline(card, fp, dates, vs)
            print(f'{split} {name}: n={len(dates)} mean qerr {q:.2f} excess {e:+.2f}; per-date qerr '
                  f'{" ".join(f"{p[0]:.2f}" for p in per)}; excess {" ".join(f"{p[1]:+.2f}" for p in per)}; '
                  f'range qerr [{min(p[0] for p in per):.2f}, {max(p[0] for p in per):.2f}] '
                  f'excess [{min(p[1] for p in per):+.2f}, {max(p[1] for p in per):+.2f}]')

    print('\n== 3. Twenty-step rollouts: t7_mx ERA5 excess % per initial date')
    for split in ('test', 'summer'):
        card, fp, dates, vs = pair(results, 't7_mx', split, steps=20)
        for step in (1, 12, 20):
            v = [s_v for s_v in vs if s_v[0] == step]
            per = [headline(card, fp, [d], v)[1] for d in dates]
            print(f'{split} lead {6 * step:3d} h: n={len(dates)} mean {np.mean(per):+.2f} '
                  f'range [{min(per):+.2f}, {max(per):+.2f}] per-date {" ".join(f"{p:+.2f}" for p in per)}')

    print('\n== 4. Distinct resamples of the non-circular 3-day block bootstrap')
    for n in (4, 8):
        starts = range(n - 3 + 1)
        need = -(-n // 3)
        combos = {tuple(sorted(sum((list(range(s, s + 3)) for s in c), [])[:n]))
                  for c in itertools.product(starts, repeat=need)}
        weight = np.zeros(n)
        for c in itertools.product(starts, repeat=need):
            for i in sum((list(range(s, s + 3)) for s in c), [])[:n]:
                weight[i] += 1
        print(f'n={n}: {len(combos)} distinct multisets; relative date weights {np.round(weight / weight.mean(), 2)}')


if __name__ == '__main__':
    main()
