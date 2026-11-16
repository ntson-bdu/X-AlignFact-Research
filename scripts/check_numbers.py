"""Final cross-check: recompute the numbers printed in the manuscript from the released outputs.

    python scripts/check_numbers.py --root $ROOT

Each line: OK / DIFF, the manuscript value, the recomputed value (rounded the way the
paper prints it), and where it appears. Exit code 1 if any DIFF. Values that cannot be
recomputed because an input file is missing are listed as MISSING. Tables 6 and 7
(probes, ERASER) are printed from results/explain_summary.csv for a manual comparison.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xalignfact.config import DEFAULT_SEEDS  # noqa: E402
from xalignfact.stats import collect  # noqa: E402

# ---- values as printed in main_v3.tex ---------------------------------------------
MAIN = {  # Table 2: (EN F1, VI F1, CLCS, joint) mean / sd
    "mono_en": {"en_test_f1": (0.737, 0.002), "vi_test_en_head_f1": (0.519, 0.025)},
    "mono_vi": {"vi_test_f1": (0.817, 0.011)},
    "mt_xlmr_vi": {"en_test_f1": (0.723, 0.007), "vi_test_f1": (0.817, 0.010),
                   "par_clcs": (0.694, 0.040), "par_joint_accuracy": (0.573, 0.039)},
    "phobert_vi": {"vi_test_f1": (0.841, 0.004)},
    "phobert_vi_noseg": {"vi_test_f1": (0.794, 0.008)},
    "full": {"en_test_f1": (0.719, 0.009), "vi_test_f1": (0.814, 0.008),
             "par_clcs": (0.873, 0.011), "par_joint_accuracy": (0.713, 0.014)},
}
MAIN_DELTA = {  # Table 2, Delta = system - X-AlignFact (points, 95% CI)
    ("mono_en", "en_test", "macro_f1"): (1.7, 1.4, 2.0),
    ("mono_vi", "vi_test", "macro_f1"): (0.3, -0.7, 1.3),
    ("mt_xlmr_vi", "en_test", "macro_f1"): (0.4, 0.2, 0.6),
    ("mt_xlmr_vi", "vi_test", "macro_f1"): (0.3, -0.5, 1.1),
    ("mt_xlmr_vi", "par_test", "clcs"): (-18.0, -20.0, -15.9),
    ("mt_xlmr_vi", "par_test", "joint_accuracy"): (-14.0, -15.9, -12.2),
    ("phobert_vi", "vi_test", "macro_f1"): (2.7, 1.3, 4.0),
}
ABL = {  # Table 3: Delta to full (EN F1, VI F1, CLCS, joint), points
    "no_cons": (-0.2, 0.4, -12.6, -11.2), "no_repr": (0.2, 0.4, 0.3, 0.0),
    "no_proto": (0.5, 0.3, -0.1, 0.0), "no_parallel": (-1.1, -1.1, -13.5, -11.6),
    "translate_train": (0.1, 0.0, -3.9, -2.1), "flat_curriculum": (0.6, 0.0, -0.7, -0.3),
    "fixed_ratio": (0.4, -0.1, 0.1, 0.3), "plus_infonce": (0.1, 1.0, 0.0, 0.2),
    "plus_infonce_labelaware": (0.3, 0.4, 0.4, 0.6), "cons_only": (0.1, 0.7, 0.1, -0.2),
    "translate_train_cons": (-0.2, -0.1, -1.3, -1.0),
}
ABL_SIG = {  # cells marked * in Table 3 (p_Holm < 0.05)
    ("no_cons", "par_test", "clcs"), ("no_cons", "par_test", "joint_accuracy"),
    ("no_proto", "en_test", "macro_f1"), ("no_parallel", "en_test", "macro_f1"),
    ("no_parallel", "par_test", "clcs"), ("no_parallel", "par_test", "joint_accuracy"),
    ("translate_train", "par_test", "clcs"), ("translate_train", "par_test", "joint_accuracy"),
    ("flat_curriculum", "en_test", "macro_f1"), ("fixed_ratio", "en_test", "macro_f1"),
    ("plus_infonce_labelaware", "en_test", "macro_f1"),
}
BM25 = {  # Table 8: VI macro-F1 with BM25 top-k (mean, sd)
    "mono_vi": [(0.607, 0.013), (0.623, 0.015), (0.577, 0.034)],
    "mt_xlmr_vi": [(0.605, 0.011), (0.648, 0.038), (0.585, 0.045)],
    "full": [(0.602, 0.008), (0.652, 0.021), (0.594, 0.019)],
}
OOD = {  # Table 5 (runs_ood): in-domain CLCS, joint; OOD CLCS, joint (mean, sd)
    "mt_xlmr_vi": [(0.663, 0.034), (0.550, 0.019), (0.684, 0.041), (0.603, 0.035)],
    "translate_train": [(0.835, 0.006), (0.689, 0.012), (0.748, 0.017), (0.651, 0.009)],
    "full": [(0.875, 0.008), (0.717, 0.004), (0.750, 0.003), (0.653, 0.005)],
}
MTQ = {  # Table 4: (CLCS, joint) per group
    "full": {"all": (0.873, 0.713), "no_flag": (0.873, 0.712), "number_mismatch": (0.771, 0.586),
             "labse_claim_Q1(low)": (0.863, 0.698), "labse_claim_Q2": (0.852, 0.693),
             "labse_claim_Q3": (0.895, 0.751), "labse_claim_Q4(high)": (0.882, 0.711)},
    "translate_train": {"all": (0.834, 0.692), "no_flag": (0.835, 0.692), "number_mismatch": (0.736, 0.557),
                        "labse_claim_Q1(low)": (0.838, 0.677), "labse_claim_Q2": (0.824, 0.662),
                        "labse_claim_Q3": (0.845, 0.742), "labse_claim_Q4(high)": (0.830, 0.689)},
    "mt_xlmr_vi": {"all": (0.694, 0.573), "no_flag": (0.687, 0.567), "number_mismatch": (0.621, 0.471),
                   "labse_claim_Q1(low)": (0.665, 0.547), "labse_claim_Q2": (0.661, 0.532),
                   "labse_claim_Q3": (0.721, 0.625), "labse_claim_Q4(high)": (0.728, 0.588)},
}
FIG4 = {  # Fig. 3 (agreement matrix) and text, % of pairs pooled over seeds
    "mt_xlmr_vi": {"agree": 69.4, "en_nei_vi_r": 17.5, "en_nei": 32.9, "vi_nei": 11.7},
    "full": {"agree": 87.3, "max_off": 3.1, "en_nei": 32.4, "vi_nei": 32.0},
}
SIZES = {"par_test": 1000, "par_test_nei": 359}

res = {"OK": 0, "DIFF": 0, "MISSING": 0}


def report(status, what, paper, got):
    res[status] += 1
    if status != "OK" or VERBOSE:
        print(f"{status:7s} {what:60s} paper={paper}  recomputed={got}")


def cmp_val(what, paper, got, nd=3):
    if got is None or (isinstance(got, float) and np.isnan(got)):
        return report("MISSING", what, paper, got)
    g = round(float(got), nd)
    report("OK" if abs(g - paper) < 0.5 * 10 ** -nd + 1e-9 else "DIFF", what, paper, f"{g:.{nd}f}")


def stat(df, v, col):
    g = df[(df.variant == v) & df.seed.isin(DEFAULT_SEEDS) & ~df.diverged.astype(bool)]
    if col not in g or g[col].dropna().empty:
        return None, None
    x = g[col].dropna().values
    return x.mean(), (x.std(ddof=1) if len(x) > 1 else np.nan)


def main(root):
    runs = collect(root)
    print("== Table 2 (main results) ==")
    for v, cols in MAIN.items():
        for c, (m, s) in cols.items():
            gm, gs = stat(runs, v, c)
            cmp_val(f"T2 {v} {c} mean", m, gm)
            cmp_val(f"T2 {v} {c} sd", s, gs)
    cp = f"{root}/results/comparisons_vs_full.csv"
    if os.path.exists(cp):
        cmp = pd.read_csv(cp)
        def row(v, t, m):
            r = cmp[(cmp.variant == v) & (cmp.set == t) & (cmp.metric == m)]
            return r.iloc[0] if len(r) else None
        for (v, t, m), (d, lo, hi) in MAIN_DELTA.items():
            r = row(v, t, m)
            for name, pv, key in (("diff", d, "diff"), ("ci_low", lo, "ci_low"), ("ci_high", hi, "ci_high")):
                cmp_val(f"T2 delta {v} {t} {m} {name}", pv, None if r is None else 100 * r[key], 1)
        print("== Table 3 (controlled study) ==")
        keys = [("en_test", "macro_f1"), ("vi_test", "macro_f1"), ("par_test", "clcs"), ("par_test", "joint_accuracy")]
        for v, vals in ABL.items():
            for (t, m), pv in zip(keys, vals):
                r = row(v, t, m)
                cmp_val(f"T3 {v} {t} {m}", pv, None if r is None else 100 * r["diff"], 1)
                if r is not None:
                    sig = r.p_holm < 0.05
                    want = (v, t, m) in ABL_SIG
                    report("OK" if sig == want else "DIFF", f"T3 star {v} {t} {m}", want, bool(sig))
    else:
        report("MISSING", "comparisons_vs_full.csv", "-", cp)
    print("== Table 8 (BM25) ==")
    for v, vals in BM25.items():
        for k, (m, s) in zip((1, 3, 5), vals):
            gm, gs = stat(runs, v, f"vi_test_bm25_k{k}_f1")
            cmp_val(f"T8 {v} k={k} mean", m, gm); cmp_val(f"T8 {v} k={k} sd", s, gs)
    print("== Table 5 (out of domain, runs_ood) ==")
    try:
        ood = collect(root, "runs_ood")
        cols = ["par_clcs", "par_joint_accuracy", "par_ood_vifc_clcs", "par_ood_vifc_joint_accuracy"]
        for v, vals in OOD.items():
            for c, (m, s) in zip(cols, vals):
                g = ood[ood.variant == v][c].dropna().values if c in ood else np.array([])
                cmp_val(f"T5 {v} {c} mean", m, g.mean() if len(g) else None)
                cmp_val(f"T5 {v} {c} sd", s, g.std(ddof=1) if len(g) > 1 else None)
    except Exception as e:  # noqa: BLE001
        report("MISSING", "runs_ood", "-", repr(e))
    print("== Table 4 (translation quality) and Fig. 3 ==")
    qp = f"{root}/data/translation_quality.csv"
    if os.path.exists(qp):
        q = pd.read_csv(qp)
        grp = {"all": np.ones(len(q), bool), "no_flag": ~(q.neg_mismatch | q.num_mismatch).values,
               "number_mismatch": q.num_mismatch.values}
        qs = pd.qcut(q["labse_claim"].rank(method="first"), 4, labels=["Q1(low)", "Q2", "Q3", "Q4(high)"])
        for lab in qs.cat.categories:
            grp[f"labse_claim_{lab}"] = (qs == lab).values
        for v, groups in MTQ.items():
            preds = [np.load(f"{root}/runs/{v}/seed{s}/preds_par_test.npz") for s in DEFAULT_SEEDS
                     if os.path.exists(f"{root}/runs/{v}/seed{s}/preds_par_test.npz")
                     and s in set(runs[(runs.variant == v) & ~runs.diverged.astype(bool)].seed)]
            for gname, (c_p, j_p) in groups.items():
                m = grp[gname]
                c = np.mean([(d["en_preds"][m] == d["vi_preds"][m]).mean() for d in preds]) if preds else None
                j = np.mean([((d["en_preds"][m] == d["labels"][m]) & (d["vi_preds"][m] == d["labels"][m])).mean()
                             for d in preds]) if preds else None
                cmp_val(f"T4 {v} {gname} CLCS", c_p, c); cmp_val(f"T4 {v} {gname} joint", j_p, j)
            if preds and v in FIG4:
                M = np.zeros((3, 3))
                for d in preds:
                    for e, w in zip(d["en_preds"], d["vi_preds"]):
                        M[int(e), int(w)] += 1
                M = 100 * M / M.sum()
                f = FIG4[v]
                cmp_val(f"Fig3 {v} agreement %", f["agree"], np.trace(M), 1)
                cmp_val(f"Fig3 {v} EN-head NEI %", f["en_nei"], M[2].sum(), 1)
                cmp_val(f"Fig3 {v} VI-head NEI %", f["vi_nei"], M[:, 2].sum(), 1)
                if "en_nei_vi_r" in f:
                    cmp_val(f"Fig3 {v} EN NEI / VI R %", f["en_nei_vi_r"], M[2, 1], 1)
                if "max_off" in f:
                    cmp_val(f"Fig3 {v} largest off-diagonal %", f["max_off"], (M - np.diag(np.diag(M))).max(), 1)
                y = preds[0]["labels"]
                cmp_val("Text: parallel test pairs", SIZES["par_test"], len(y), 0)
                cmp_val("Text: gold NEI share % (35.9)", 35.9, 100 * np.mean(y == 2), 1)
    else:
        report("MISSING", "translation_quality.csv", "-", qp)
    print("== Tables 6-7 (probes, ERASER): compare by hand ==")
    ep = f"{root}/results/explain_summary.csv"
    if os.path.exists(ep):
        print(pd.read_csv(ep).to_string())
    else:
        report("MISSING", "explain_summary.csv", "-", ep)
    print(f"\nSummary: {res}")
    return res["DIFF"] == 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--verbose", action="store_true", help="also print the OK lines")
    a = ap.parse_args()
    VERBOSE = a.verbose
    sys.exit(0 if main(a.root) else 1)
