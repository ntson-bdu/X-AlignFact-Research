"""Part 5 - Aggregation over seeds and statistical tests (AE2, R1.2, AE5a).

Reported for every variant: mean +- SD over seeds and a 95% t-interval.
Comparisons with the reference variant (default "full") use a paired bootstrap
over test examples of the seed-averaged difference:
    for each resample of test indices: d = mean_s [ m(variant_s) - m(ref_s) ]
Two-sided p-values are Holm-corrected across all compared variants.
Only seeds present for both variants are used.
"""
from __future__ import annotations

import glob
import os
from typing import Callable, Dict, List

import numpy as np
import pandas as pd
from scipy import stats as sps
from .utils import load_json


def _macro_f1(y, p, k: int = 3):
    """Fast macro-F1 (same definition as sklearn, zero_division=0)."""
    cm = np.bincount(y * k + p, minlength=k * k).reshape(k, k)
    tp = np.diag(cm).astype(float)
    denom = cm.sum(0) + cm.sum(1)
    f1 = np.divide(2 * tp, denom, out=np.zeros(k), where=denom > 0)
    return float(f1.mean())


CHANCE_F1 = 0.20   # constant prediction on 3 balanced classes gives macro-F1 ~= 0.167


def _diverged(log_path: str) -> bool:
    """A run is 'diverged' if, after epoch 0, every monitored dev F1 fell to the level of a
    constant predictor (training collapse). Rule fixed before looking at test results."""
    if not os.path.exists(log_path):
        return False
    log = load_json(log_path)
    for e in log[1:]:
        f1s = [e[k] for k in ("vi_dev_f1", "en_dev_f1") if k in e]
        if f1s and max(f1s) < CHANCE_F1:
            return True
    return False


def collect(root: str, runs_subdir: str = "runs") -> pd.DataFrame:
    rows = []
    for path in glob.glob(f"{root}/{runs_subdir}/*/seed*/results.json"):
        r = load_json(path)
        t = r["test"]
        row = {"variant": r["variant"], "seed": r["seed"], "best_epoch": r["best_epoch"],
               "diverged": _diverged(os.path.join(os.path.dirname(path), "train_log.json"))}
        for tag, m in t.items():
            if tag.startswith("par_"):
                pre = "par" if tag == "par_test" else tag
                for k in ("clcs", "clcs_p", "kappa", "joint_accuracy"):
                    row[f"{pre}_{k}"] = m[k]
            else:
                row[f"{tag}_f1"] = m["macro_f1"]
                row[f"{tag}_acc"] = m["accuracy"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["variant", "seed"]).reset_index(drop=True)


def summarise(df: pd.DataFrame, metrics: List[str], exclude_diverged: bool = True) -> pd.DataFrame:
    """Mean ± SD over seeds. Diverged runs are counted in `n_diverged` and, by default,
    excluded from the statistics (report both in the paper)."""
    out = []
    for v, g0 in df.groupby("variant"):
        n_div = int(g0.get("diverged", pd.Series(False, index=g0.index)).sum())
        g = g0[~g0["diverged"]] if (exclude_diverged and "diverged" in g0) else g0
        row = {"variant": v, "n_seeds": len(g), "n_diverged": n_div}
        for m in metrics:
            x = g[m].dropna().values
            if len(x) == 0:
                continue
            mean, sd = x.mean(), x.std(ddof=1) if len(x) > 1 else float("nan")
            half = sps.t.ppf(0.975, len(x) - 1) * sd / np.sqrt(len(x)) if len(x) > 1 else float("nan")
            row[m] = f"{mean:.4f} ± {sd:.4f}"
            row[f"{m}_mean"], row[f"{m}_sd"] = mean, sd
            row[f"{m}_ci95"] = (mean - half, mean + half)
        out.append(row)
    return pd.DataFrame(out)


def _load_preds(root, variant, seed, tag):
    p = f"{root}/runs/{variant}/seed{seed}/preds_{tag}.npz"
    return np.load(p) if os.path.exists(p) else None


def paired_bootstrap(root: str, variant: str, ref: str, tag: str, seeds: List[int],
                     metric: str = "macro_f1", n_boot: int = 10000, rng_seed: int = 0) -> Dict:
    """tag: 'en_test', 'vi_test', ... (macro-F1) or 'par_test' (metric='clcs' or 'joint_accuracy')."""
    pairs = []
    div = _diverged_seeds(root, [variant, ref])
    for s in seeds:
        if s in div:
            continue
        a, b = _load_preds(root, variant, s, tag), _load_preds(root, ref, s, tag)
        if a is not None and b is not None:
            pairs.append((a, b))
    if not pairs:
        return {}
    y = pairs[0][0]["labels"]
    n = len(y)

    if tag == "par_test":
        def m(d, idx):
            e, v, yy = d["en_preds"][idx], d["vi_preds"][idx], d["labels"][idx]
            if metric == "joint_accuracy":
                return float(((e == yy) & (v == yy)).mean())
            return float((e == v).mean())
    else:
        def m(d, idx):
            return _macro_f1(d["labels"][idx], d["preds"][idx])

    full_idx = np.arange(n)
    observed = np.mean([m(a, full_idx) - m(b, full_idx) for a, b in pairs])
    rng = np.random.default_rng(rng_seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = np.mean([m(a, idx) - m(b, idx) for a, b in pairs])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {"variant": variant, "ref": ref, "set": tag, "metric": metric, "n_seeds": len(pairs),
            "diff": float(observed), "ci_low": float(lo), "ci_high": float(hi), "p": float(min(p, 1.0))}


def _diverged_seeds(root: str, variants: List[str]) -> set:
    bad = set()
    for v in variants:
        for p in glob.glob(f"{root}/runs/{v}/seed*/train_log.json"):
            if _diverged(p):
                bad.add(int(os.path.basename(os.path.dirname(p))[4:]))
    return bad


def holm(pvals: List[float]) -> List[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(running, 1.0)
    return adj.tolist()


def compare_all(root: str, variants: List[str], ref: str = "full",
                seeds: List[int] | None = None, n_boot: int = 10000) -> pd.DataFrame:
    from .config import DEFAULT_SEEDS

    seeds = seeds or DEFAULT_SEEDS
    plan = [("vi_test", "macro_f1"), ("en_test", "macro_f1"),
            ("par_test", "clcs"), ("par_test", "joint_accuracy")]
    rows = []
    for v in variants:
        if v == ref:
            continue
        for tag, metric in plan:
            r = paired_bootstrap(root, v, ref, tag, seeds, metric, n_boot)
            if r:
                rows.append(r)
    df = pd.DataFrame(rows)
    if len(df):
        df["p_holm"] = np.nan
        for (tag, metric), g in df.groupby(["set", "metric"]):
            df.loc[g.index, "p_holm"] = holm(g["p"].tolist())
    return df
