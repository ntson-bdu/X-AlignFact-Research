"""EXP-6 baseline: PhoBERT-base fine-tuned on ViFactCheck with the same splits, seeds,
selection rule (VI-dev macro-F1), epochs, patience and examples per step as `mono_vi`.

PhoBERT expects word-segmented Vietnamese. The original PhoBERT paper uses the
VnCoreNLP RDRSegmenter; this script uses `underthesea.word_tokenize(format="text")`
by default. [AUTHOR CHECK] State the segmenter in the paper, or switch to VnCoreNLP.

python scripts/baseline_phobert.py --root ... --seeds 42 123 456 789 2024
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # before torch is imported
import argparse
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xalignfact.config import DEFAULT_SEEDS, make_config  # noqa: E402
from xalignfact.data import load_splits  # noqa: E402
from xalignfact.evaluate import classification_metrics  # noqa: E402
from xalignfact.sampler import StreamIterator  # noqa: E402
from xalignfact.utils import Logger, env_info, save_json, seed_everything  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--seeds", nargs="*", type=int, default=DEFAULT_SEEDS)
ap.add_argument("--model", default="vinai/phobert-base")
ap.add_argument("--segmenter", default="underthesea", choices=["underthesea", "none"])
ap.add_argument("--max-length", type=int, default=256)
a = ap.parse_args()


def segment(texts):
    if a.segmenter == "none":
        return list(texts)
    from underthesea import word_tokenize

    return [word_tokenize(t, format="text") for t in texts]


def encode(tok, df):
    enc = tok(segment(df["claim"]), segment(df["evidence"]), truncation="longest_first",
              max_length=a.max_length, padding="max_length", return_tensors="pt")
    return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
            "labels": torch.tensor(df["label"].values)}


@torch.no_grad()
def predict(model, d, device, bs=128):
    model.eval()
    probs = []
    for i in range(0, len(d["labels"]), bs):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            lg = model(input_ids=d["input_ids"][i:i + bs].to(device),
                       attention_mask=d["attention_mask"][i:i + bs].to(device)).logits
        probs.append(torch.softmax(lg.float(), -1).cpu())
    p = torch.cat(probs)
    return {"labels": d["labels"].numpy(), "preds": p.argmax(-1).numpy(), "probs": p.numpy()}


from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402

base = make_config("mono_vi", root=a.root)
tok = AutoTokenizer.from_pretrained(a.model)
splits = load_splits(base)
cache = f"{base.data_dir}/tensors/phobert_{a.segmenter}.pt"
if os.path.exists(cache):
    data = torch.load(cache, weights_only=False)
else:
    data = {k: encode(tok, splits[k]) for k in ("vi_train", "vi_dev", "vi_test")}
    torch.save(data, cache)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for seed in a.seeds:
    cfg = make_config("mono_vi", seed=seed, root=a.root)
    vname = "phobert_vi" if a.segmenter == "underthesea" else "phobert_vi_noseg"
    run_dir = f"{a.root}/runs/{vname}/seed{seed}"
    if os.path.exists(f"{run_dir}/results.json"):
        print(f"[skip] phobert seed {seed}")
        continue
    os.makedirs(run_dir, exist_ok=True)
    log = Logger(f"{run_dir}/train.log")
    save_json(env_info(), f"{run_dir}/env.json")
    seed_everything(seed)
    model = AutoModelForSequenceClassification.from_pretrained(a.model, num_labels=3).to(device)
    steps = math.ceil(len(data["vi_train"]["labels"]) / 32)
    bs = cfg.step_budget                       # same examples per step as mono_vi
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=cfg.num_epochs * steps,
                                                pct_start=cfg.warmup_ratio, anneal_strategy="cos")
    it = StreamIterator(len(data["vi_train"]["labels"]), seed * 1000 + 1)
    best, best_state, best_ep, bad, hist = -1, None, -1, 0, []
    for ep in range(cfg.num_epochs):
        model.train()
        for _ in range(steps):
            i = it.take(bs)
            d = data["vi_train"]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                out = model(input_ids=d["input_ids"][i].to(device),
                            attention_mask=d["attention_mask"][i].to(device),
                            labels=d["labels"][i].to(device))
            opt.zero_grad(set_to_none=True)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            sched.step()
        p = predict(model, data["vi_dev"], device)
        f1 = classification_metrics(p["labels"], p["preds"])["macro_f1"]
        hist.append({"epoch": ep, "vi_dev_f1": f1})
        log(f"[phobert seed {seed} ep {ep}] vi_dev_f1={f1:.4f}")
        if f1 > best:
            best, best_ep, bad = f1, ep, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    model.load_state_dict(best_state)
    p = predict(model, data["vi_test"], device)
    np.savez_compressed(f"{run_dir}/preds_vi_test.npz", **p)
    save_json(hist, f"{run_dir}/train_log.json")
    save_json({"variant": vname, "seed": seed, "best_epoch": best_ep,
               "best_selection_score": best, "segmenter": a.segmenter,
               "test": {"vi_test": classification_metrics(p["labels"], p["preds"])}},
              f"{run_dir}/results.json")
    del model
    torch.cuda.empty_cache()
