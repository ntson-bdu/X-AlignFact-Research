"""Parts 3-5 CLI: train/evaluate variants over seeds (resume-safe: finished runs are skipped).

python scripts/run.py --root /content/drive/MyDrive/xalignfact_v2 --variants full --seeds 42 123 456 789 2024
python scripts/run.py --root ... --group ablations
python scripts/run.py --root ... --group baselines
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # before torch is imported
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xalignfact.config import DEFAULT_SEEDS, VARIANTS, make_config  # noqa: E402
from xalignfact.train import run  # noqa: E402

GROUPS = {
    "main": ["full"],
    "ablations": ["no_cons", "no_repr", "no_proto", "no_parallel", "translate_train",
                  "plus_infonce", "flat_curriculum", "fixed_ratio"],
    "optional": ["plus_infonce_labelaware", "shared_head"],
    "followup": ["cons_only", "translate_train_cons"],
    "baselines": ["mono_en", "mono_vi", "mt_xlmr", "mt_xlmr_vi"],
}

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--variants", nargs="*", default=[])
ap.add_argument("--group", choices=sorted(GROUPS), default=None)
ap.add_argument("--seeds", nargs="*", type=int, default=DEFAULT_SEEDS)
ap.add_argument("--save-best", action="store_true", help="save best.pt (~1.1 GB per run)")
ap.add_argument("--save-phase-ckpts", action="store_true", help="for the gradient analysis")
ap.add_argument("--overwrite", action="store_true")
a = ap.parse_args()

variants = list(a.variants) + (GROUPS[a.group] if a.group else [])
if not variants:
    sys.exit(f"Nothing to run. Variants: {sorted(VARIANTS)}; groups: {sorted(GROUPS)}")
for v in variants:
    for s in a.seeds:
        cfg = make_config(v, seed=s, root=a.root, save_best_checkpoint=a.save_best,
                          save_phase_checkpoints=a.save_phase_ckpts)
        run(cfg, overwrite=a.overwrite)
