"""Supplementary tables from existing outputs (no training).

    python scripts/make_supp_tables.py --root $ROOT

Writes LaTeX tables to results/supp/ (paste or \\input into supplementary.tex):
  tabS_perclass.tex     accuracy, macro-F1 and per-class F1 of every system (VI and EN test)
  tabS_mtxlmr.tex       MT-XLM-R selected with the combined rule vs. on Vietnamese F1
  tabS_confirm.tex      runs of the confirmatory InfoNCE comparison, one row per seed
  tabS_ood_examples.tex out-of-domain pairs: native Vietnamese claim and its VinAI translation
Every value is mean +- SD over the default seeds, diverged runs excluded (same rule as the paper).
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xalignfact.config import DEFAULT_SEEDS  # noqa: E402
from xalignfact.stats import collect  # noqa: E402

LAB = ["S", "R", "NEI"]
NAMES = {"mono_en": "XLM-R (EN only)", "mono_vi": "XLM-R (VI only)", "mt_xlmr_vi": "MT-XLM-R",
         "mt_xlmr": "MT-XLM-R (combined rule)", "phobert_vi": "PhoBERT-base (VI only)",
         "full": "X-AlignFact", "plus_infonce": "+ InfoNCE"}
CONFIRM_SEEDS = [3407, 11, 17, 29, 101]


def f1s(y, p):
    out = []
    for c in range(3):
        tp = np.sum((p == c) & (y == c)); fp = np.sum((p == c) & (y != c)); fn = np.sum((p != c) & (y == c))
        out.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return np.array(out), float(np.mean(p == y))


def ok_seeds(per_run, v):
    g = per_run[(per_run.variant == v) & per_run.seed.isin(DEFAULT_SEEDS) & ~per_run.diverged.astype(bool)]
    return sorted(g.seed.tolist())


def ms(x, d=3):
    x = np.asarray(x, float)
    return f"${x.mean():.{d}f} \\pm {x.std(ddof=1):.{d}f}$" if len(x) > 1 else (f"${x.mean():.{d}f}$" if len(x) else "--")


def tab_perclass(root, per_run, out):
    rows = []
    # which prediction file fills the VI / EN columns (None = not applicable)
    plan = {"mono_en": ("vi_test_en_head", "en_test"),   # zero-shot: English head on VI inputs
            "mono_vi": ("vi_test", None), "phobert_vi": ("vi_test", None),
            "mt_xlmr_vi": ("vi_test", "en_test"), "full": ("vi_test", "en_test")}
    for v, tags in plan.items():
        seeds = ok_seeds(per_run, v)
        cells = []
        for tag in tags:
            if tag is None:
                cells += ["--"] * 5
                continue
            accs, f1m, pc = [], [], []
            for s in seeds:
                p = f"{root}/runs/{v}/seed{s}/preds_{tag}.npz"
                if not os.path.exists(p):
                    continue
                d = np.load(p)
                f, a = f1s(d["labels"], d["preds"])
                accs.append(a); f1m.append(f.mean()); pc.append(f)
            if not accs:
                cells += ["--"] * 5
                continue
            pc = np.array(pc)
            cells += [ms(accs), ms(f1m)] + [ms(pc[:, c]) for c in range(3)]
        rows.append(f"{NAMES[v]} ({len(seeds)}) & " + " & ".join(cells) + r" \\")
    tex = r"""\begin{table*}[h]
\centering
\caption{Accuracy, macro-F1 and per-class F1 on the test sets (mean $\pm$ SD over seeds; number of seeds in parentheses, diverged runs excluded; Vietnamese columns of XLM-R (EN only): zero-shot, English head on Vietnamese inputs). S/R/NEI: \textsc{Supports}/\textsc{Refutes}/\textsc{Not Enough Info}.}
\label{tab:perclass}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lcccccccccc}
\hline
 & \multicolumn{5}{c}{Vietnamese test (ViFactCheck)} & \multicolumn{5}{c}{English test (NLI-FEVER dev)} \\
System & Acc. & Macro-F1 & F1$_S$ & F1$_R$ & F1$_{NEI}$ & Acc. & Macro-F1 & F1$_S$ & F1$_R$ & F1$_{NEI}$ \\
\hline
""" + "\n".join(rows) + r"""
\hline
\end{tabular}}
\end{table*}
"""
    open(f"{out}/tabS_perclass.tex", "w").write(tex)


def tab_mtxlmr(root, per_run, out):
    cmp_p = f"{root}/results/comparisons_vs_full.csv"
    cmp = pd.read_csv(cmp_p) if os.path.exists(cmp_p) else pd.DataFrame()
    rows = []
    for v in ("mt_xlmr", "mt_xlmr_vi", "full"):
        g = per_run[(per_run.variant == v) & per_run.seed.isin(DEFAULT_SEEDS)]
        div = int(g.diverged.astype(bool).sum())
        g = g[~g.diverged.astype(bool)]
        if g.empty:
            continue
        cells = [ms(g[c]) for c in ("en_test_f1", "vi_test_f1", "par_clcs", "par_joint_accuracy")]
        if v != "full" and len(cmp):
            for tag, met in (("vi_test", "macro_f1"), ("par_test", "clcs")):
                r = cmp[(cmp.variant == v) & (cmp.set == tag) & (cmp.metric == met)]
                if len(r):
                    r = r.iloc[0]
                    star = "^{*}" if r.p_holm < 0.05 else ""
                    cells.append(f"${100*r['diff']:+.1f}\\,[{100*r.ci_low:.1f}, {100*r.ci_high:.1f}]{star}$")
                else:
                    cells.append("--")
        else:
            cells += ["--", "--"]
        rows.append(f"{NAMES[v]} & {len(g)} ({div}) & " + " & ".join(cells) + r" \\")
    tex = r"""\begin{table*}[h]
\centering
\caption{MT-XLM-R with the two selection rules: the combined rule of the article ($0.7\,\mathrm{F1}_{\mathrm{VI}}^{\mathrm{dev}} + 0.3\,\mathrm{CLCS}^{\mathrm{dev}}$) and Vietnamese development F1 (the version in the main table). Mean $\pm$ SD; seeds used (diverged runs excluded); $\Delta$ to X-AlignFact in points with 95\% paired-bootstrap interval, * significant after Holm correction.}
\label{tab:mtxlmr}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lccccccc}
\hline
System & Seeds (div.) & EN F1 & VI F1 & CLCS & Joint acc. & $\Delta$ VI F1 & $\Delta$ CLCS \\
\hline
""" + "\n".join(rows) + r"""
\hline
\end{tabular}}
\end{table*}
"""
    open(f"{out}/tabS_mtxlmr.tex", "w").write(tex)


def tab_confirm(per_run, out):
    g = per_run[per_run.seed.isin(CONFIRM_SEEDS) & per_run.variant.isin(["full", "plus_infonce"])]
    g = g.sort_values(["seed", "variant"])
    rows = []
    for _, r in g.iterrows():
        vals = [f"{r[c]:.3f}" if pd.notna(r.get(c)) else "--"
                for c in ("en_test_f1", "vi_test_f1", "par_clcs", "par_joint_accuracy")]
        rows.append(f"{int(r.seed)} & {NAMES[r.variant]} & " + " & ".join(vals)
                    + f" & {'yes' if bool(r.diverged) else 'no'}" + r" \\")
    tex = r"""\begin{table}[h]
\centering
\caption{Confirmatory InfoNCE comparison on five additional seeds (one row per run). Diverged runs, defined as in Section V-F of the article, are excluded from the comparison reported there.}
\label{tab:confirm}
\begin{tabular}{rlccccc}
\hline
Seed & System & EN F1 & VI F1 & CLCS & Joint & Diverged \\
\hline
""" + "\n".join(rows) + r"""
\hline
\end{tabular}
\end{table}
"""
    open(f"{out}/tabS_confirm.tex", "w").write(tex)


def tab_ood_examples(root, out, n=5, seed=0):
    tr_p, vi_p = f"{root}/data/vifc_test_vi2en.csv", f"{root}/data/splits/vi_test.jsonl"
    if not (os.path.exists(tr_p) and os.path.exists(vi_p)):
        print("  ood examples: missing", tr_p if not os.path.exists(tr_p) else vi_p)
        return
    tr = pd.read_csv(tr_p)
    vi = pd.read_json(vi_p, lines=True)
    idx = np.random.default_rng(seed).choice(len(tr), size=n, replace=False)
    esc = lambda t: " ".join(str(t).split()).replace("\\", "").replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("#", r"\#").replace("$", r"\$")
    rows = []
    for i in sorted(idx):
        i = int(tr.idx.iloc[i]) if "idx" in tr else int(i)
        lab = LAB[int(vi.label.iloc[i])] if "label" in vi else "--"
        rows.append(f"{i} & \\vi{{{esc(vi.claim.iloc[i])}}} & {esc(tr.claim_en.iloc[i])} & {lab}" + r" \\")
    tex = r"""\begin{table*}[h]
\centering
\caption{Out-of-domain parallel pairs: native ViFactCheck test claims and their VinAI translations (five pairs drawn at random with a fixed seed; claims only, evidence omitted).}
\label{tab:oodex}
\begin{tabular}{rp{0.42\textwidth}p{0.38\textwidth}c}
\hline
Index & Vietnamese claim (original) & English claim (VinAI Translate) & Label \\
\hline
""" + "\n".join(rows) + r"""
\hline
\end{tabular}
\end{table*}
"""
    open(f"{out}/tabS_ood_examples.tex", "w", encoding="utf-8").write(tex)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    a = ap.parse_args()
    out = f"{a.root}/results/supp"
    os.makedirs(out, exist_ok=True)
    per_run = collect(a.root)
    for name, fn in (("perclass", lambda: tab_perclass(a.root, per_run, out)),
                     ("mtxlmr", lambda: tab_mtxlmr(a.root, per_run, out)),
                     ("confirm", lambda: tab_confirm(per_run, out)),
                     ("ood_examples", lambda: tab_ood_examples(a.root, out))):
        fn()
        print(f"  {name}: done")
    for f in sorted(os.listdir(out)):
        print("----", f)
        print(open(f"{out}/{f}", encoding="utf-8").read())
