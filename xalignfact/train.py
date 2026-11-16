"""Part 3 - One training pipeline for every variant (main model, ablations, baselines).

    results = run(make_config("full", seed=42))

Per run, written to cfg.run_dir:
    config.json, env.json, train_log.json, results.json,
    preds_<set>.npz (labels, preds, probs) for every evaluation set,
    best.pt (optional), phase checkpoints (optional), train.log
Model selection uses dev data only; test sets are evaluated once, after training.
"""
from __future__ import annotations

import contextlib
import math
import os
import time
from typing import Dict, Optional

import numpy as np
import torch

from .config import Config
from .data import extra_eval_sets, extra_parallel_sets, load_tensors
from .evaluate import clcs_metrics, classification_metrics, predict, predict_parallel
from .model import EN, VI, XAlignFact
from .sampler import CurriculumSampler
from .utils import Logger, env_info, load_json, save_json, seed_everything

TRAIN_SETS = ["en_train", "vi_train", "par_train"]
DEV_SETS = ["en_dev", "vi_dev", "par_dev"]
TEST_SETS = ["en_test", "vi_test", "par_test"]


def _amp(cfg: Config, device: torch.device):
    if cfg.bf16 and device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def build_batch(data: Dict[str, Dict], picks: Dict[str, torch.Tensor], device) -> Dict:
    b = {}
    if "en" in picks:
        d = data["en_train"]
        i = picks["en"]
        b.update(en_ids=d["input_ids"][i].to(device), en_mask=d["attention_mask"][i].to(device),
                 en_y=d["labels"][i].to(device))
    if "vi" in picks:
        d = data["vi_train"]
        i = picks["vi"]
        b.update(vi_ids=d["input_ids"][i].to(device), vi_mask=d["attention_mask"][i].to(device),
                 vi_y=d["labels"][i].to(device))
    if "par" in picks:
        d = data["par_train"]
        i = picks["par"]
        b.update(par_en_ids=d["en_input_ids"][i].to(device), par_en_mask=d["en_attention_mask"][i].to(device),
                 par_vi_ids=d["vi_input_ids"][i].to(device), par_vi_mask=d["vi_attention_mask"][i].to(device),
                 par_y=d["labels"][i].to(device))
    return b


def selection_score(cfg: Config, dev: Dict[str, float]) -> float:
    if cfg.selection == "vi_f1":
        return dev["vi_dev_f1"]
    if cfg.selection == "en_f1":
        return dev["en_dev_f1"]
    w = cfg.selection_vi_weight
    return w * dev["vi_dev_f1"] + (1 - w) * dev["par_dev_clcs"]


@torch.no_grad()
def evaluate_dev(model, data, cfg, device) -> Dict[str, float]:
    out = {}
    with _amp(cfg, device):
        if cfg.use_en or cfg.selection == "en_f1":
            p = predict(model, data["en_dev"], EN, device, cfg.eval_batch_size)
            out["en_dev_f1"] = classification_metrics(p["labels"], p["preds"])["macro_f1"]
        p = predict(model, data["vi_dev"], VI, device, cfg.eval_batch_size)
        out["vi_dev_f1"] = classification_metrics(p["labels"], p["preds"])["macro_f1"]
        pp = predict_parallel(model, data["par_dev"], device, cfg.eval_batch_size)
        out["par_dev_clcs"] = clcs_metrics(pp)["clcs"]
    return out


def run(cfg: Config, overwrite: bool = False, encoder_module=None,
        device: Optional[torch.device] = None) -> Dict:
    os.makedirs(cfg.run_dir, exist_ok=True)
    res_path = f"{cfg.run_dir}/results.json"
    if os.path.exists(res_path) and not overwrite:
        print(f"[skip] {cfg.variant}/seed{cfg.seed}: results.json exists")
        return load_json(res_path)

    log = Logger(f"{cfg.run_dir}/train.log")
    cfg.to_json(f"{cfg.run_dir}/config.json")
    save_json(env_info(), f"{cfg.run_dir}/env.json")
    seed_everything(cfg.seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    manifest = load_json(f"{cfg.data_dir}/manifest.json")
    data = load_tensors(cfg, TRAIN_SETS + DEV_SETS + TEST_SETS + extra_eval_sets(cfg)
                       + extra_parallel_sets(cfg))
    model = XAlignFact(cfg, vocab_size=manifest.get("vocab_size"),
                       encoder_module=encoder_module).to(device)

    sizes = {"en": len(data["en_train"]["labels"]) if cfg.use_en else 0,
             "vi": len(data["vi_train"]["labels"]) if cfg.use_vi else 0,
             "par": len(data["par_train"]["labels"]) if cfg.use_parallel else 0}
    sampler = CurriculumSampler(cfg, sizes, cfg.seed)
    steps = cfg.steps_per_epoch or math.ceil(len(data["vi_train"]["labels"]) / 32)
    # Labelled-language streams; translate-train pairs join the EN/VI pools, keeping the
    # task-loss scale identical to `full`.
    n_streams = max(1, int(cfg.use_en) + int(cfg.use_vi))

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.lr, total_steps=cfg.num_epochs * steps,
        pct_start=cfg.warmup_ratio, anneal_strategy="cos")

    log(f"[run] {cfg.variant} seed={cfg.seed} steps/epoch={steps} sizes={sizes}")
    history, best, best_state, best_epoch, bad = [], -1.0, None, -1, 0
    for epoch in range(cfg.num_epochs):
        model.train()
        t0 = time.time()
        counts = dict(zip(CurriculumSampler.STREAMS, sampler.counts(epoch)))
        sums: Dict[str, float] = {}
        for _ in range(steps):
            picks = sampler.next(epoch)
            batch = build_batch(data, picks, device)
            with _amp(cfg, device):
                losses = model.training_losses(batch, n_streams)
            opt.zero_grad(set_to_none=True)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            sched.step()
            for k, v in losses.items():
                sums[k] = sums.get(k, 0.0) + float(v.detach())
        model.eval()
        dev = evaluate_dev(model, data, cfg, device)
        score = selection_score(cfg, dev)
        entry = {"epoch": epoch, "phase": cfg.phase_for_epoch(epoch)["name"],
                 "composition": counts, "lr": sched.get_last_lr()[0],
                 "time_sec": round(time.time() - t0, 1),
                 **{f"train_{k}": v / steps for k, v in sums.items()},
                 **dev, "selection_score": score}
        history.append(entry)
        log(f"[ep {epoch:2d}] {entry['phase']} {counts} loss={entry['train_total']:.4f} "
            + " ".join(f"{k}={v:.4f}" for k, v in dev.items()) + f" sel={score:.4f}")

        if cfg.save_phase_checkpoints and epoch == cfg.phase_for_epoch(epoch)["end_epoch"]:
            torch.save(model.state_dict(), f"{cfg.run_dir}/ckpt_{entry['phase']}_ep{epoch}.pt")
        if score > best:
            best, best_epoch, bad = score, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                log(f"[run] early stop at epoch {epoch} (best epoch {best_epoch})")
                break
    save_json(history, f"{cfg.run_dir}/train_log.json")

    # ---- test evaluation with the dev-selected checkpoint --------------------
    model.load_state_dict(best_state)
    model.eval()
    if cfg.save_best_checkpoint:
        torch.save(best_state, f"{cfg.run_dir}/best.pt")
    results = {"variant": cfg.variant, "seed": cfg.seed, "best_epoch": best_epoch,
               "best_selection_score": best, "epochs_run": len(history), "test": {}}
    with torch.no_grad(), _amp(cfg, device):
        eval_plan = [("en_test", EN, "en_test"), ("vi_test", VI, "vi_test"),
                     ("vi_test", EN, "vi_test_en_head")]       # zero-shot style read-out
        eval_plan += [(n, VI, n) for n in extra_eval_sets(cfg)]  # retrieved evidence (EXP-5)
        for set_name, head, tag in eval_plan:
            if set_name not in data:
                continue
            p = predict(model, data[set_name], head, device, cfg.eval_batch_size)
            np.savez_compressed(f"{cfg.run_dir}/preds_{tag}.npz", **p)
            results["test"][tag] = classification_metrics(p["labels"], p["preds"])
        pp = predict_parallel(model, data["par_test"], device, cfg.eval_batch_size)
        np.savez_compressed(f"{cfg.run_dir}/preds_par_test.npz", **pp)
        results["test"]["par_test"] = clcs_metrics(pp)
        for name in extra_parallel_sets(cfg):            # EXP-9: out-of-domain consistency
            if name in data:
                pp = predict_parallel(model, data[name], device, cfg.eval_batch_size)
                np.savez_compressed(f"{cfg.run_dir}/preds_{name}.npz", **pp)
                results["test"][name] = clcs_metrics(pp)
    save_json(results, res_path)
    log(f"[run] done: EN={results['test']['en_test']['macro_f1']:.4f} "
        f"VI={results['test']['vi_test']['macro_f1']:.4f} "
        f"CLCS={results['test']['par_test']['clcs']:.4f} (best epoch {best_epoch})")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results
