"""Part 5 CLI: tables with mean ± SD over seeds and paired-bootstrap comparisons.

python scripts/aggregate.py --root /content/drive/MyDrive/xalignfact_v2
Writes results/summary.csv, results/comparisons_vs_full.csv, results/per_run.csv
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xalignfact.config import DEFAULT_SEEDS  # noqa: E402
from xalignfact.stats import collect, compare_all, summarise  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--ref", default="full")
ap.add_argument("--n-boot", type=int, default=10000)
a = ap.parse_args()

out = f"{a.root}/results"
os.makedirs(out, exist_ok=True)
df_all = collect(a.root)
df_all.to_csv(f"{out}/per_run.csv", index=False)
metrics = [c for c in df_all.columns if c.endswith("_f1") or c.startswith("par_")]
# Paper tables use the five pre-specified seeds only; extra (confirmatory) seeds are
# summarised separately so that they never change the headline numbers.
df = df_all[df_all["seed"].isin(DEFAULT_SEEDS)]
summ = summarise(df, metrics)
summ.to_csv(f"{out}/summary.csv", index=False)
extra = df_all[~df_all["seed"].isin(DEFAULT_SEEDS)]
if len(extra):
    summarise(extra, metrics).to_csv(f"{out}/summary_extra_seeds.csv", index=False)
    summarise(df_all, metrics).to_csv(f"{out}/summary_all_seeds.csv", index=False)
    print(f"(extra seeds found for {sorted(extra['variant'].unique())}: "
          "see summary_extra_seeds.csv / summary_all_seeds.csv)")
show = ["variant", "n_seeds", "n_diverged"] + [m for m in ("en_test_f1", "vi_test_f1", "vi_test_en_head_f1",
                                             "par_clcs", "par_joint_accuracy", "par_kappa")
                                 if m in summ.columns]
print(summ[show].to_string(index=False))
div = df_all[df_all["diverged"]]
if len(div):
    print("\nDiverged runs (excluded from mean/SD and from paired tests):")
    print(div[["variant", "seed", "best_epoch"]].to_string(index=False))

# diagnostic runs (e.g. PhoBERT without word segmentation) are summarised but kept out of
# the Holm family, so they cannot change the corrected p-values reported in the paper
variants = sorted(v for v in df["variant"].unique() if not v.endswith("_noseg"))
if a.ref in variants:
    cmp = compare_all(a.root, variants, ref=a.ref, n_boot=a.n_boot)
    cmp.to_csv(f"{out}/comparisons_vs_{a.ref}.csv", index=False)
    if len(cmp):
        print(cmp.round(4).to_string(index=False))
