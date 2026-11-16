"""Part 1 CLI: build leakage-free splits, tokenise, and build the BM25 test sets.

python scripts/prepare_data.py --root /content/drive/MyDrive/xalignfact_v2 \
    --legacy-parallel /content/drive/MyDrive/xalignfact/data/parallel/parallel_en_vi.json
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xalignfact.config import make_config  # noqa: E402
from xalignfact.data import build_retrieved_vi_test, build_splits, tokenize_all  # noqa: E402
from xalignfact.utils import Logger  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--legacy-parallel", default="", help="v1 parallel_en_vi.json to reuse (recommended)")
ap.add_argument("--claim-field", default="auto", choices=["auto", "premise", "hypothesis"])
ap.add_argument("--skip-bm25", action="store_true")
a = ap.parse_args()

cfg = make_config("full", root=a.root, parallel_legacy_path=a.legacy_parallel,
                  fever_claim_field=a.claim_field)
os.makedirs(cfg.data_dir, exist_ok=True)
log = Logger(f"{cfg.data_dir}/prepare.log")
splits = build_splits(cfg, log)
tokenize_all(cfg, splits, log)
if not a.skip_bm25:
    build_retrieved_vi_test(cfg, log)
log("[prepare] done")
