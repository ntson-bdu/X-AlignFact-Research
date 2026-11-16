"""Part 6 CLI: descriptive analyses on the `full` model.

Requires runs made with --save-best (and --save-phase-ckpts for gradients):
python scripts/run.py --root ... --variants full --seeds 42 --save-best --save-phase-ckpts --overwrite
python scripts/run_analysis.py --root ... --seed 42
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # before torch is imported
import argparse
import glob
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xalignfact.analysis import (attention_figure, error_analysis, export_translation_sheet,  # noqa: E402
                                 gradient_cosines, load_model, tsne_plot)
from xalignfact.config import make_config  # noqa: E402
from xalignfact.data import get_tokenizer, load_tensors  # noqa: E402
from xalignfact.model import EN, VI  # noqa: E402
from xalignfact.utils import load_json, save_json  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--variant", default="full")
a = ap.parse_args()

cfg = make_config(a.variant, seed=a.seed, root=a.root)
out = f"{a.root}/results/analysis_{a.variant}_seed{a.seed}"
os.makedirs(out, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab = load_json(f"{cfg.data_dir}/manifest.json")["vocab_size"]
data = load_tensors(cfg, ["en_train", "vi_train", "par_train", "en_test", "vi_test"])

# Error analysis (all errors counted; examples exported for reading)
for tag, split in (("vi_test", "vi_test"), ("en_test", "en_test")):
    d = f"{out}/errors_{tag}"
    os.makedirs(d, exist_ok=True)
    rep = error_analysis(f"{cfg.run_dir}/preds_{tag}.npz", f"{cfg.data_dir}/splits/{split}.jsonl", d)
    print(tag, rep["pairs"])

# Translation annotation sheet (EXP-4)
print("annotation sheet:", export_translation_sheet(cfg))

best = f"{cfg.run_dir}/best.pt"
if os.path.exists(best):
    model = load_model(cfg, best, vocab, device)
    tsne_plot(model, data["en_test"], data["vi_test"], f"{out}/tsne_test.png", device)
    tok = get_tokenizer(cfg)
    attention_figure(model, data["vi_test"], [0, 1, 2], tok, VI, f"{out}/attention_vi.png", device)
    attention_figure(model, data["en_test"], [0, 1, 2], tok, EN, f"{out}/attention_en.png", device)
    del model
else:
    print("best.pt not found: skipping t-SNE and attention figures")

ckpts = {}
for ph in cfg.phases:
    hits = glob.glob(f"{cfg.run_dir}/ckpt_{ph['name']}_ep*.pt")
    if hits:
        ckpts[ph["name"]] = hits[0]
if ckpts:
    g = gradient_cosines(cfg, ckpts, data, vocab, device)
    save_json(g, f"{out}/gradient_cosines.json")
    print(g)
else:
    print("no phase checkpoints: skipping gradient analysis")
