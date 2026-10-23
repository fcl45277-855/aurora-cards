"""Pre-registered test-split checks for the mechanism cards (the experiment log, test criteria 7-8).

  python scripts/decide_test.py [--split test] [--results <dir>]
"""
import argparse
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UP4 = ("z500", "t850", "u850", "v850")
UP5 = UP4 + ("q700",)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--results", default=str(ROOT / "results"), help="directory with the <card>__<split>_h2 summaries")
    a = ap.parse_args()

    def L(n):
        p = Path(a.results) / f"{n}__{a.split}_h2" / "summary.json"
        return json.loads(p.read_text()) if p.exists() else None

    rr = lambda s, v, t=1: s["per_var"][f"{v}_s{t}"]["rmse_ref"]
    b, q = L("bf16"), L("q0_sim")
    ex = lambda s, v: rr(s, v) ** 2 - rr(b, v) ** 2
    m = lambda s, ref: statistics.mean(ex(s, v) / ex(ref, v) for v in UP5)
    out = []
    m01, m02 = L("m01_q0_decoder_only"), L("m02_q0_no_decoder")
    if m01 and m02:
        S1 = {v: (rr(m01, v) / rr(q, v)) ** 2 for v in UP4}
        S2 = {v: (rr(m02, v) / rr(q, v)) ** 2 for v in UP4}
        med = statistics.median(S1.values())
        sums = {v: round(S1[v] + S2[v], 3) for v in UP4}
        out.append(f"7  m01 4-var median share {med:.3f} (ref >=0.8: {'yes' if med >= 0.8 else 'no'}); m01+m02 {sums} "
                   f"(ref [0.85,1.05]: {'yes' if all(0.85 <= x <= 1.05 for x in sums.values()) else 'no'})")
    ma, mw, mc = L("m01a_dec_act_only"), L("m01w_dec_weight_only"), L("m01c_dec_bf16_ctl")
    if ma and mw:
        r = ma["qerr_pct"] / mw["qerr_pct"]
        out.append(f"8a m01a/m01w {r:.2f} (>=3: {'PASS' if r >= 3 else 'FAIL'})")
    if mc:
        out.append(f"8b m01c {mc['qerr_pct']:.2f} (<=5: {'PASS' if mc['qerr_pct'] <= 5 else 'FAIL'})")
    kv = L("m1_to_kv")
    if kv and m01:
        sh = (kv["qerr_pct"] / m01["qerr_pct"]) ** 2
        out.append(f"8c m1_to_kv share of m01 {sh:.3f} (>=0.6: {'PASS' if sh >= 0.6 else 'FAIL'})")
    act, K, K5 = L("m1kv_act"), L("m1kv_K"), L("m1kv_K_a05")
    if act and K:
        sk = m(K, act)
        out.append(f"8d S_K {sk:.3f} (>=0.7: {'PASS' if sk >= 0.7 else 'FAIL'})")
    if K and K5:
        rat = m(K5, K)
        slope = statistics.mean(math.log2(math.sqrt(ex(K, v) / ex(K5, v))) for v in UP5)
        out.append(f"8e K_a05 / K excess MSE {rat:.4f} (<=0.15: {'PASS' if rat <= 0.15 else 'FAIL'}; linear 0.25), slope {slope:.2f}")
    kc = L("m1kv_kc")
    if kc and kv:
        cut = 1 - m(kc, kv)
        out.append(f"8f K centring cut {100 * cut:.1f}% (>=68%: {'PASS' if cut >= 0.68 else 'FAIL'})")
    g, gn = L("m1kv_g02"), L("m1kv_g02_noLR")
    if g and gn:
        ok = g["qerr_pct"] <= 30 and gn["qerr_pct"] >= 2 * g["qerr_pct"]
        out.append(f"8g m1kv_g02 {g['qerr_pct']:.2f}, noLR {gn['qerr_pct']:.2f} ({gn['qerr_pct'] / g['qerr_pct']:.1f}x) ({'PASS' if ok else 'FAIL'})")
    g02, t1, t8, t4 = L("g02_svdq_a08_r32"), L("t1_g02_dec_bf16"), L("t8_g02_kv_bf16"), L("t4_g02_dec_fp8")
    if g02 and t1 and t8:
        gap = g02["qerr_pct"] - t1["qerr_pct"]
        d = g02["qerr_pct"] - t8["qerr_pct"]
        out.append(f"8h t8 cut {d:.2f} vs 25% of g02-T1 gap {0.25 * gap:.2f} ({'PASS' if d < 0.25 * gap else 'FAIL'})")
    if t4 and t1:
        out.append(f"8i t4 {t4['qerr_pct']:.2f} <= T1+2 {t1['qerr_pct'] + 2:.2f}: {'PASS' if t4['qerr_pct'] <= t1['qerr_pct'] + 2 else 'FAIL'}")
    us = {k: L(f"u_g02_dec16_but_{k}") for k in ("to_out", "fc2", "to_kv")}
    if t1 and all(us.values()):
        inc = {k: s["qerr_pct"] ** 2 - t1["qerr_pct"] ** 2 for k, s in us.items()}
        top = max(inc, key=inc.get)
        out.append(f"8k leave-one-in increments (qerr^2 over T1) {dict((k, round(v, 1)) for k, v in inc.items())}: "
                   f"largest {top} ({'PASS' if top == 'to_out' else 'FAIL'})")
    print("\n".join(out))


if __name__ == "__main__":
    main()
