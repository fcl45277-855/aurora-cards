"""Extended analyses from existing records (CPU only): paired recipe differences, rollout intervals, FP32 absolute
skill, spectra at long lead and per-date robustness of the attention-replay predictor.

  A. Paired differences on the 30-date test month (6 and 12 h): card A minus card B in qerr and ERA5 excess, with the
     same 3-day non-circular block resamples applied to both cards (2000 resamples, seed 0, percentile intervals).
  B. Twenty-step rollouts (test month, the initial dates shared with FP32): mean ERA5 excess over the nine variables
     at 6, 12, 24, 48, 72, 96 and 120 h, with 3-day block-bootstrap intervals over initial dates.
  C. FP32 absolute RMSE against ERA5 at 12, 24, 72 and 120 h for z500, t850, 2t, 10u and msl (root of the mean MSE
     over initial dates; z500 in m2 s-2, t850 and 2t in K, 10u in m s-1, msl in Pa).
  D. Zonal power spectra (20-70 deg) at 24, 72 and 120 h from _diag_fields_*: card / FP32 power per wavenumber bin and
     the small-scale ratio (wavenumber >= 64), averaged over initial dates; FP32 / ERA5 for reference.
  E. Attention-replay predictor: Spearman correlation between each replay date's predictor and the module damage
     (squared-qerr excess over BF16 on the 12 development dates), with exact one-sided permutation p-values
     (all n! orderings), per model variant; also against the damage of the replay date alone.

    python scripts/extended_analyses.py [--results <dir>] [--pairs t7_mx:c08_t7set ...] > extended_analyses.txt

The results directory holds the per-card records.jsonl of the full result archive (docs/DATA.md); it defaults to
$AQ_RESULTS, else <repo>/results. Model variants for E are read from <dir>, <dir>/../results_finetuned and
<dir>/../results_small when present.
"""
import argparse
import glob
import itertools
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VARS = ["2t", "10u", "10v", "msl", "z500", "t850", "u850", "v850", "q700"]
MODS = ["perc_enc", "perc_dec", "en0", "en1", "en2", "de0", "de1", "de2"]
STAGE = {"en0": "enc0", "en1": "enc1", "en2": "enc2", "de0": "dec0", "de1": "dec1", "de2": "dec2"}


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


def resample(dates, rng, L=3):
    n = len(dates)
    out = []
    while len(out) < n:
        s = int(rng.integers(0, n - L + 1))
        out.extend(dates[s + i] for i in range(L))
    return out[:n]


def headline(card, fp, dates, cells):
    ks = [(d, s, v) for d in dates for (s, v) in cells]
    f = np.array([fp[k]['rmse_truth'] for k in ks])
    return (100 * float(np.mean(np.array([card[k]['rmse_ref'] for k in ks]) / f)),
            100 * float(np.mean(np.array([card[k]['rmse_truth'] for k in ks]) / f - 1)))


def common(fp, *cards, steps=None):
    keys = set(fp)
    for c in cards:
        keys &= set(c)
    if steps is not None:
        keys = {k for k in keys if k[1] in steps}
    return sorted({k[0] for k in keys}), sorted({(k[1], k[2]) for k in keys})


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def perm_p(x, y):
    """Exact one-sided p of Spearman rho >= observed over all orderings of y (no ties expected)."""
    obs = spearman(x, y)
    rx = np.argsort(np.argsort(x)).astype(float)
    ge = tot = 0
    for p in itertools.permutations(range(len(y))):
        tot += 1
        ge += float(np.corrcoef(rx, np.array(p, float))[0, 1]) >= obs - 1e-12
    return obs, ge / tot


def section_a(results, pairs, n=2000):
    print('== A. Paired differences, test month 6 and 12 h (A - B; 3-day non-circular block bootstrap, 2000 resamples)')
    fp = load(find(results, 'fp32__test_h2'))
    print(f"{'A':22s} {'B':22s} {'dqerr':>7s} {'95% interval':>17s} {'dexcess %':>9s} {'95% interval':>17s} {'P(dexc<0)':>9s}")
    for pa in pairs:
        a, b = pa.split(':')
        try:
            ca, cb = load(find(results, f'{a}__test_h2')), load(find(results, f'{b}__test_h2'))
        except FileNotFoundError as e:
            print(f'{a:22s} {b:22s} missing {e}')
            continue
        dates, cells = common(fp, ca, cb)
        qa, ea = headline(ca, fp, dates, cells)
        qb, eb = headline(cb, fp, dates, cells)
        rng = np.random.default_rng(0)
        d, xs = [], []
        for _ in range(n):
            s = resample(dates, rng)
            xa, xb = headline(ca, fp, s, cells), headline(cb, fp, s, cells)
            d.append((xa[0] - xb[0], xa[1] - xb[1]))
            xs.append(xa)
        d, xs = np.array(d), np.array(xs)
        iq, ie = np.percentile(d[:, 0], [2.5, 97.5]), np.percentile(d[:, 1], [2.5, 97.5])
        aq, ae = np.percentile(xs[:, 0], [2.5, 97.5]), np.percentile(xs[:, 1], [2.5, 97.5])
        print(f'{a:22s} {b:22s} {qa - qb:+7.2f} [{iq[0]:+6.2f}, {iq[1]:+6.2f}] {ea - eb:+9.3f} [{ie[0]:+6.3f}, {ie[1]:+6.3f}] '
              f'{float(np.mean(d[:, 1] < 0)):9.3f}   (n={len(dates)} dates; A {qa:.2f} [{aq[0]:.2f}, {aq[1]:.2f}] / '
              f'{ea:+.2f} [{ae[0]:+.2f}, {ae[1]:+.2f}], B {qb:.2f}/{eb:+.2f})')
        worst = max(((s_, v) for (s_, v) in cells), key=lambda k: np.mean([ca[(dd, k[0], k[1])]['rmse_truth'] / fp[(dd, k[0], k[1])]['rmse_truth'] for dd in dates]))
        wv = 100 * (np.mean([ca[(dd, worst[0], worst[1])]['rmse_truth'] / fp[(dd, worst[0], worst[1])]['rmse_truth'] for dd in dates]) - 1)
        print(f'{"":45s} worst cell of A: {worst[1]} step {worst[0]} {wv:+.2f} %')


def section_b(results, cards, n=2000):
    print('\n== B. Twenty-step rollouts, test month: mean ERA5 excess % over nine variables (3-day block bootstrap over dates)')
    fp = load(find(results, 'fp32__test_h20'))
    leads = [1, 2, 4, 8, 12, 16, 20]
    for c in cards:
        try:
            cc = load(find(results, f'{c}__test_h20'))
        except FileNotFoundError:
            print(f'{c}: no test_h20 records')
            continue
        dates, _ = common(fp, cc)
        rng = np.random.default_rng(0)
        idx = [resample(dates, rng) for _ in range(n)]
        row = []
        for s in leads:
            cells = [(s, v) for v in VARS]
            m = headline(cc, fp, dates, cells)[1]
            b = np.array([headline(cc, fp, r, cells)[1] for r in idx])
            lo, hi = np.percentile(b, [2.5, 97.5])
            row.append(f'{6 * s:4d} h {m:+6.2f} [{lo:+6.2f}, {hi:+6.2f}]')
        per = [100 * float(np.mean([cc[(d, 20, v)]['rmse_truth'] / fp[(d, 20, v)]['rmse_truth'] - 1 for d in dates])) for v in VARS]
        print(f'{c} (n={len(dates)} initial dates)')
        print('   ' + '\n   '.join(row))
        print('   120 h per variable: ' + ' '.join(f'{v} {p:+.2f}' for v, p in zip(VARS, per)))


def section_c(results):
    print('\n== C. FP32 absolute RMSE against ERA5, test month (root mean square over initial dates)')
    fp = load(find(results, 'fp32__test_h20'))
    dates = sorted({k[0] for k in fp})
    print(f'n = {len(dates)} initial dates ({dates[0]}..{dates[-1]}, 06 UTC initialisation)')
    print(f"{'var':6s} " + ' '.join(f'{h:>9d} h' for h in (12, 24, 72, 120)))
    for v in ['z500', 't850', '2t', '10u', 'msl']:
        vals = [float(np.sqrt(np.mean([fp[(d, h // 6, v)]['rmse_truth'] ** 2 for d in dates]))) for h in (12, 24, 72, 120)]
        print(f'{v:6s} ' + ' '.join(f'{x:11.3f}' for x in vals))


def section_d(results):
    print('\n== D. Zonal power spectra (20-70 deg), card / FP32 per wavenumber bin, mean over initial dates')
    base = results / 'results' if (results / 'results').exists() else results
    for f in sorted(glob.glob(str(base / '_diag_fields_test_h20*.json'))):
        if f.endswith('_maps.npz'):
            continue
        d = json.load(open(f))
        if 'recipes' not in d or 'k_edges' not in d:
            continue
        edges = d['k_edges']
        small = [i for i in range(len(edges) - 1) if edges[i] >= 64]
        print(f'-- {Path(f).name}: dates {d["dates"][0]}..{d["dates"][-1]} (n={len(d["dates"])}); bins {edges}; small scale = k >= 64')
        for step in ('4', '12', '20'):
            if step not in d['fp32']:
                continue
            for v in d['spec_fields']:
                fp = np.array([d['fp32'][step][x]['spectra'][v] for x in d['dates']])
                tr = np.array([d['truth'][step][x]['spectra'][v] for x in d['dates']])
                lines = [f'   {6 * int(step):3d} h {v:5s} fp32/era5 small {np.mean(fp[:, small].sum(1) / tr[:, small].sum(1)):.3f}']
                for name, r in d['recipes'].items():
                    if step not in r['per_date']:
                        continue
                    cs = np.array([r['per_date'][step][x]['spectra'][v] for x in d['dates'] if x in r['per_date'][step]])
                    ratio = np.mean(cs / fp[:len(cs)], 0)
                    sm = np.mean(cs[:, small].sum(1) / fp[:len(cs), small].sum(1))
                    lines.append(f'{name} small {sm:.3f} bins ' + ' '.join(f'{x:.2f}' for x in ratio))
                print(' | '.join(lines))


def qerr_by_date(results, name, fp):
    c = load(find(results, f'{name}__dev_h2'))
    dates, cells = common(fp, c)
    return {d: headline(c, fp, [d], cells)[0] for d in dates}, headline(c, fp, dates, cells)[0]


def section_e(results):
    print('\n== E. Attention-replay predictor err_in_flip: Spearman rho with exact one-sided permutation p (n = 8 modules)')
    for tag, res in (('pretrained', results), ('finetuned', results.parent / 'results_finetuned'), ('small', results.parent / 'results_small')):
        if not (res / '_replay_swin.json').exists():
            continue
        fp = load(find(res, 'fp32__dev_h2'))
        qb_d, qb = qerr_by_date(res, 'bf16', fp)
        tgt_d, tgt = {}, {}
        for m in MODS:
            qd, q = qerr_by_date(res, f'b_attn_{m}', fp)
            tgt[m] = q ** 2 - qb ** 2
            tgt_d[m] = {d: qd[d] ** 2 - qb_d[d] ** 2 for d in qd}
        swin = json.load(open(res / '_replay_swin.json'))
        dec = json.load(open(res / '_replay_attn_decoder.json'))
        enc = json.load(open(res / '_replay_attn_encoder.json'))
        dates = list(swin['dates'])
        preds = {}
        for d in dates:
            p = {}
            for m in MODS:
                s = swin['dates'][d][STAGE[m]] if m in STAGE else (dec if m == 'perc_dec' else enc)['dates'][d]
                p[m] = s['alpha_1.0']['share_err_in_flipped']
            preds[d] = p
        t = np.array([tgt[m] for m in MODS])
        print(f'{tag}: replay dates {dates}')
        for d in dates:
            x = np.array([preds[d][m] for m in MODS])
            r, p = perm_p(x, t)
            td = np.array([tgt_d[m].get(d, np.nan) for m in MODS])
            r2, p2 = perm_p(x, td) if not np.isnan(td).any() else (np.nan, np.nan)
            print(f'   {d}: vs 12-date damage rho {r:+.2f} (p = {p:.4f}); vs same-date damage rho {r2:+.2f} (p = {p2:.4f})')
        xm = np.array([np.mean([preds[d][m] for d in dates]) for m in MODS])
        r, p = perm_p(xm, t)
        print(f'   mean over replay dates: rho {r:+.2f} (p = {p:.4f})')
        sw = [m for m in MODS if m in STAGE]
        r6, p6 = perm_p(np.array([preds[dates[0]][m] for m in sw]), np.array([tgt[m] for m in sw]))
        print(f'   six Swin stages only, first date: rho {r6:+.2f} (p = {p6:.4f}, n = 6)')
        be = res / '_b_eval.json'
        if be.exists():
            pe = json.load(open(be))['pred']
            if all('params' in pe[m] for m in MODS):
                rp, pp = perm_p(np.array([pe[m]['params'] for m in MODS], float), t)
                print(f'   parameter count of the attention projections: rho {rp:+.2f} (p = {pp:.4f})')
        print('   damage per date (rank order of modules is stable if the rows agree): ' +
              '; '.join(f"{d}: " + ' > '.join(sorted(MODS, key=lambda m: -tgt_d[m][d])[:3]) for d in sorted(tgt_d[MODS[0]])[:12:4]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default=os.environ.get('AQ_RESULTS', str(ROOT / 'results')))
    ap.add_argument('--pairs', nargs='*', default=['t7_mx:c08_t7set', 't7_a16:c08_t7set', 't7_a16:t7_mx',
                                                    'u8_w8a8_mx:t7_a16', 'u8_w8a16_mx:t7_a16', 'u8_w8a8_ch:t7_a16'])
    ap.add_argument('--rollouts', nargs='*', default=['t7_mx', 't7_a16', 'c08_t7set', 'c08_weight_only_gptq', 'u8_w8a8_mx', 'q0_sim'])
    ap.add_argument('--sections', default='ABCDE')
    a = ap.parse_args()
    res = Path(a.results)
    for s, f in (('A', lambda: section_a(res, a.pairs)), ('B', lambda: section_b(res, a.rollouts)), ('C', lambda: section_c(res)),
                 ('D', lambda: section_d(res)), ('E', lambda: section_e(res))):
        if s in a.sections:
            f()


if __name__ == '__main__':
    main()
