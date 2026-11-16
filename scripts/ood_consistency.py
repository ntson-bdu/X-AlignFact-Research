"""EXP-9 (reviewer question: is the CLCS gain specific to the translated-FEVER distribution?).

Consistency is re-measured on ViFactCheck test (native Vietnamese news, gold evidence) paired with
its machine translation into English. No system sees this domain in English during training, and
the VI side is native text, not a translation.

Step 1 (GPU, ~10-20 min with VinAI Translate; or --mt google, rate-limited):
    python scripts/ood_consistency.py --root $ROOT --translate
Step 2 (GPU, re-trains each system into runs_ood/ so the headline runs stay untouched):
    python scripts/ood_consistency.py --root $ROOT --train --save-best --variants full mt_xlmr_vi translate_train
    (--save-best keeps best.pt for full and mt_xlmr_vi, seeds 42/123/456, for EXP-10)
Step 3 (CPU): tables + paired bootstrap vs full
    python scripts/ood_consistency.py --root $ROOT --report
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import glob
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xalignfact.config import DEFAULT_SEEDS, make_config  # noqa: E402
from xalignfact.stats import collect, holm  # noqa: E402

SET = "par_ood_vifc"


def _load(root, sub, v, s, name):
    p = f"{root}/{sub}/{v}/seed{s}/preds_{name}.npz"
    return np.load(p) if os.path.exists(p) else None


def _metric(d, idx, metric):
    e, v, y = d["en_preds"][idx], d["vi_preds"][idx], d["labels"][idx]
    return (e == v).mean() if metric == "clcs" else ((e == y) & (v == y)).mean()


def bootstrap(root, sub, variant, ref, name, metric, seeds, n_boot=10000, rng_seed=0):
    pairs = [(a, b) for s in seeds
             if (a := _load(root, sub, variant, s, name)) is not None
             and (b := _load(root, sub, ref, s, name)) is not None]
    if not pairs:
        return None
    n = len(pairs[0][0]["labels"])
    full_idx = np.arange(n)
    obs = np.mean([_metric(a, full_idx, metric) - _metric(b, full_idx, metric) for a, b in pairs])
    rng = np.random.default_rng(rng_seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = np.mean([_metric(a, idx, metric) - _metric(b, idx, metric) for a, b in pairs])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))
    return {"variant": variant, "set": name, "metric": metric, "n_seeds": len(pairs),
            "diff": obs, "ci_low": lo, "ci_high": hi, "p": p}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--translate", action="store_true")
    ap.add_argument("--mt", choices=["vinai", "google"], default="vinai",
                    help="vinai: offline VinAI Translate on GPU (default); google: deep-translator")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--variants", nargs="*", default=["full", "mt_xlmr_vi", "translate_train", "cons_only"])
    ap.add_argument("--seeds", nargs="*", type=int, default=DEFAULT_SEEDS)
    ap.add_argument("--save-best", action="store_true",
                    help="keep best.pt (~1.1 GB each) for EXP-10 explainability")
    ap.add_argument("--save-best-variants", nargs="*", default=["full", "mt_xlmr_vi"])
    ap.add_argument("--save-best-seeds", nargs="*", type=int, default=[42, 123, 456])
    a = ap.parse_args()

    if a.translate:
        from xalignfact.data import build_vifc_parallel_test
        build_vifc_parallel_test(make_config("full", seed=42, root=a.root), mt=a.mt)

    if a.train:
        from xalignfact.train import run
        assert os.path.exists(f"{a.root}/data/tensors/{SET}.pt"), "run --translate first"
        for v in a.variants:
            for s in a.seeds:
                keep = a.save_best and v in a.save_best_variants and s in a.save_best_seeds
                run(make_config(v, seed=s, root=a.root, runs_subdir="runs_ood",
                                save_best_checkpoint=keep))

    if a.report:
        out = f"{a.root}/results"
        os.makedirs(out, exist_ok=True)
        if not glob.glob(f"{a.root}/runs_ood/*/seed*/results.json"):
            sys.exit("no runs in runs_ood/ yet: run --translate and --train first")
        df = collect(a.root, "runs_ood")
        cols = ["en_test_f1", "vi_test_f1", "par_clcs", "par_joint_accuracy",
                f"{SET}_clcs", f"{SET}_joint_accuracy", f"{SET}_kappa"]
        cols = [c for c in cols if c in df.columns]
        summ = df.groupby("variant")[cols].agg(["mean", "std"]).round(4)
        summ.to_csv(f"{out}/ood_summary.csv")
        print("== runs_ood: mean (std) over seeds ==")
        print(summ.to_string())
        # reproducibility check: re-runs vs headline runs on the in-domain metrics
        orig = collect(a.root, "runs")
        orig = orig[orig["seed"].isin(a.seeds) & orig["variant"].isin(a.variants)]
        rep = (df.groupby("variant")[["vi_test_f1", "par_clcs"]].mean()
               - orig.groupby("variant")[["vi_test_f1", "par_clcs"]].mean()).round(4)
        print("\n== re-run minus headline run (in-domain; should be close to 0) ==")
        print(rep.to_string())
        rows = []
        for v in a.variants:
            if v == "full":
                continue
            for name in (SET, "par_test"):
                for metric in ("clcs", "joint_accuracy"):
                    r = bootstrap(a.root, "runs_ood", v, "full", name, metric, a.seeds)
                    if r:
                        rows.append(r)
        cmp = pd.DataFrame(rows)
        if len(cmp):
            cmp["p_holm"] = np.nan
            for _, g in cmp.groupby(["set", "metric"]):
                cmp.loc[g.index, "p_holm"] = holm(g["p"].tolist())
            cmp.to_csv(f"{out}/ood_comparisons_vs_full.csv", index=False)
            print("\n== paired bootstrap vs full (runs_ood) ==")
            print(cmp.round(4).to_string(index=False))
