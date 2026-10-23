"""Apply the pre-registered decision rules to finished dev cards.

  python scripts/decide.py [--results <dir>]     # prints whichever decisions have all their cards finished

Shares are in MSE terms: S = (rmse_ref(card) / rmse_ref(ref_card))**2 per variable-step.
"""
import argparse
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
UPPER = ("z500", "t850", "u850", "v850")
VARS = ("2t", "10u", "10v", "msl", "z500", "t850", "u850", "v850", "q700")


def load(name, split="dev", steps=2):
    p = RESULTS / f"{name}__{split}_h{steps}" / "summary.json"
    return json.loads(p.read_text()) if p.exists() else None


def rr(s, var, step):
    return s["per_var"][f"{var}_s{step}"]["rmse_ref"]


def ex(s, var, step):
    """Per variable-step ERA5 excess vs FP32 (%), needs the fp32 card."""
    f = load("fp32")
    return 100 * (s["per_var"][f"{var}_s{step}"]["rmse_truth"] / f["per_var"][f"{var}_s{step}"]["rmse_truth"] - 1)


def U(s):
    """Upper-air mean qerr (%): rmse_ref(card) / rmse_truth(fp32), over z500/t850/u850/v850 x 2 steps."""
    f = load("fp32")
    return statistics.mean(100 * rr(s, v, t) / f["per_var"][f"{v}_s{t}"]["rmse_truth"] for v in UPPER for t in (1, 2))


def share(a, b, var, step):
    return (rr(a, var, step) / rr(b, var, step)) ** 2


def line(*xs):
    print("  " + " ".join(str(x) for x in xs))


def m01():
    m, q, b = load("m01_q0_decoder_only"), load("q0_sim"), load("bf16")
    if not (m and q and b):
        return
    print("## 1. m01 decoder-only (share of Q0 MSE)")
    gate = {v: abs(rr(m, v, 1) / rr(b, v, 1) - 1) for v in ("2t", "10u", "10v", "msl")}
    line("gate step-1 surface |rel diff vs bf16|:", {k: f"{x:.1e}" for k, x in gate.items()},
         "PASS" if max(gate.values()) <= 1e-6 else "FAIL -> check harness first")
    S = {f"{v}_s{t}": share(m, q, v, t) for v in VARS for t in (1, 2)}
    line("S per var-step:", {k: round(x, 3) for k, x in S.items()})
    med = statistics.median(S[f"{v}_s1"] for v in UPPER)
    line(f"median upper-air step-1 S = {med:.3f}; z500_s1 rmse_ref = {rr(m, 'z500', 1):.1f}; msl_s2 = {rr(m, 'msl', 2):.1f} Pa")
    if med >= 0.5 and rr(m, "msl", 2) >= 10:
        verdict = "PASS: high-altitude damage enters via decoder path, leaks to surface at step 2"
    elif med < 0.2:
        verdict = "REJECT: upper-air sensitivity is backbone-intrinsic"
    else:
        verdict = "IN BETWEEN: engineering note only"
    line("->", verdict)
    m2 = load("m02_q0_no_decoder")
    if m2:
        add = {v: share(m, q, v, 1) + share(m2, q, v, 1) for v in UPPER}
        line("additivity m01+m02 shares (step 1):", {k: round(x, 3) for k, x in add.items()})


def g02_rescue():
    g, c08 = load("g02_svdq_a08_r32"), load("c08_weight_only_gptq")
    if not (g and c08):
        return
    gap = g["qerr_pct"] - c08["qerr_pct"]
    print(f"## 2. rescue on g02 (qerr {g['qerr_pct']:.2f}, gap to c08 G = {gap:.2f}, upper-air mean qerr U = {U(g):.1f})")
    for name in ("t1_g02_dec_bf16", "t4_g02_dec_fp8", "t2_g02_mlpfc_fp8"):
        s = load(name)
        if not s:
            line(name, "pending")
            continue
        closed = (g["qerr_pct"] - s["qerr_pct"]) / gap
        up = {v: round(100 * (1 - share(s, g, v, 1)), 1) for v in UPPER}
        worst_ex = max(ex(s, v, t) for v in VARS for t in (1, 2))
        line(f"{name}: qerr {s['qerr_pct']:.2f}, closes {100 * closed:.0f}% of G, U {U(s):.1f}, ERA5 excess {s['excess_pct']:+.2f}%, "
             f"worst var-step excess {worst_ex:+.2f}%, upper-air MSE cut vs g02 (step 1, %) {up}")
        if name == "t1_g02_dec_bf16":
            # doc 05 wrote "U <= 25 (cut >= 40%)" from U(g02) = 42.0; this script's U(g02) is 45.8, so use the relative form
            line("   T1 -> whole-decoder precision is the answer (U cut >= 40%):", "YES" if U(s) <= 0.6 * U(g) else "NO")
        if name == "t4_g02_dec_fp8":
            line("   T4 deployable baseline (all var-step excess <= +2%):", "YES" if worst_ex <= 2 else "NO")
        if name == "t2_g02_mlpfc_fp8":
            line("   geo/token masks:", "dead (closes <40% of G)" if closed < 0.4 else
                 ("worth coding only if T1 cuts upper air <20%" if closed >= 0.5 else "marginal"))


def fold():
    g = load("g02_svdq_a08_r32")
    if not g:
        return
    print("## 3. G-fold / T3 on g02")
    for name in ("gfold_g02", "t3_g02_posemb_bf16"):
        s = load(name)
        if not s:
            line(name, "pending")
            continue
        ci = s.get("qerr_reduction_vs_q0_pct_ci95")
        line(f"{name}: qerr {s['qerr_pct']:.2f} (g02 {g['qerr_pct']:.2f}, change {100 * (1 - s['qerr_pct'] / g['qerr_pct']):+.1f}%), "
             f"z500 rmse_ref s1/s2 {rr(s, 'z500', 1):.1f}/{rr(s, 'z500', 2):.1f} vs g02 {rr(g, 'z500', 1):.1f}/{rr(g, 'z500', 2):.1f}")
    line("(the z500 offset itself needs the per-date bias_ref, which records.jsonl does not store; read the cut in z500 rmse_ref)")


def closure():
    t0, t1, t2 = load("c05_gptq_full"), load("d06_w4a8_gptq_norot") or load("d01_w4a8_norot"), load("c05_dec_fp8")
    if not (t0 and t2):
        return
    print("## 4. FP8 decoder closure on T0 = c05_gptq_full")
    if not t1:
        line("T1' pending")
        return
    q = lambda s: s["qerr_pct"] ** 2
    c = (q(t0) - q(t2)) / (q(t0) - q(t1))
    line(f"T0 {t0['qerr_pct']:.2f}, T1' ({t1['name']}) {t1['qerr_pct']:.2f}, T2' {t2['qerr_pct']:.2f}: closure = {c:.3f}")
    sd = load("s_dec")
    if sd:
        line(f"s_dec qerr {sd['qerr_pct']:.2f}")
    line("->", ">=0.6: damage sits in one place, report the finding" if c >= 0.6 else
         ("<0.2: damage in backbone, look at s_benc*/s_bdec*" if c < 0.2 else "in between"))


def sens():
    q, rows = load("q0_sim"), []
    if not q:
        return
    for g in ("dec", "attn", "mlp", "mod", "enc", "benc0", "benc1", "benc2", "bdec0", "bdec1", "bdec2", "time"):
        s = load(f"s_{g}")
        if s:
            rows.append((g, s["qerr_pct"], statistics.median(share(s, q, v, 1) for v in UPPER)))
    if rows:
        print("## sensitivity groups (only this group quantized as Q0)")
        for g, qe, sh in rows:
            line(f"s_{g}: qerr {qe:.2f}, upper-air step-1 share of Q0 MSE {sh:.3f}")


UP5 = ("z500", "t850", "u850", "v850", "q700")


def exmse(s, b, var, step=1):
    """Excess MSE over the bf16 floor (rmse_ref^2 - rmse_ref(bf16)^2)."""
    return rr(s, var, step) ** 2 - rr(b, var, step) ** 2


def tokv():
    """E2 (Q0 numerics) and E3 (g02 components); upper-air step 1, 5 variables."""
    b, full, act = load("bf16"), load("m1_to_kv"), load("m1kv_act")
    if not (b and full and act):
        return
    print("## to_kv mechanism (06 plan E2/E3; excess MSE over bf16, step 1, 5 upper-air vars)")
    m = lambda s: statistics.mean(exmse(s, b, v) / exmse(act, b, v) for v in UP5)
    K, V, w = load("m1kv_K"), load("m1kv_V"), load("m1kv_w")
    if K and V:
        sk, sv = m(K), m(V)
        path = "K path" if sk >= 0.7 else ("V path" if sv >= 0.6 and sk <= 0.25 else "mixed")
        line(f"S_K {sk:.3f}, S_V {sv:.3f}, S_K+S_V {sk + sv:.3f} (gate [0.8, 1.2]) -> {path}")
    a05, a025 = load("m1kv_K_a05"), load("m1kv_K_a025")
    if K and a05 and a025:
        import math
        p = lambda hi, lo: statistics.mean(math.log2(math.sqrt(max(exmse(hi, b, v), 1e-30) / max(exmse(lo, b, v), 1e-30))) for v in UP5)
        line(f"K amplitude slopes: p(1 vs 1/2) {p(K, a05):.2f}, p(1/2 vs 1/4) {p(a05, a025):.2f}  (1 = linear; >=1.3 = nonlinear switching)")
    if w:
        inter = statistics.mean((exmse(full, b, v) - exmse(act, b, v) - exmse(w, b, v)) / exmse(w, b, v) for v in UP5)
        line(f"W x A interaction I / ex(w) = {inter:.2f}  (>= 1: first-order picture incomplete)")
    f8 = load("m1kv_fp8")
    if f8:
        line(f"FP8 activations: qerr {f8['qerr_pct']:.2f} vs NVFP4 {act['qerr_pct']:.2f}")
    kc = load("m1kv_kc")
    if kc:
        cut = statistics.mean(1 - exmse(kc, b, v) / exmse(full, b, v) for v in UP5)
        line(f"K centring (W4A4): qerr {kc['qerr_pct']:.2f} vs m1_to_kv {full['qerr_pct']:.2f}; upper-air excess MSE cut {100 * cut:.0f}%"
             + (f" (needs >= {70 * m(K):.0f}% = 0.7 x S_K)" if K else ""))
    g = load("m1kv_g02")
    if g:
        line(f"E3: m1kv_g02 {g['qerr_pct']:.2f} (<= 30: g02 numerics neutralise to_kv)"
             + "".join(f"; {n} {load(n)['qerr_pct']:.2f} ({load(n)['qerr_pct'] / g['qerr_pct']:.1f}x)"
                       for n in ("m1kv_g02_noLR", "m1kv_g02_noSm", "m1kv_norot") if load(n)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(RESULTS), help="directory with the <card>__dev_h2/summary.json files")
    RESULTS = Path(ap.parse_args().results)
    for f in (m01, g02_rescue, fold, closure, sens, tokv):
        f()
