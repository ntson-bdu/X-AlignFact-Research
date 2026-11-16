"""Part 6 - Descriptive analyses (EXP-4, EXP-7). None of these is used as causal evidence.

Fixes vs. v1:
  * t-SNE on *test* examples; prototypes are embedded in the same t-SNE fit.
  * Error analysis counts all errors from the confusion matrix (no cap of 10);
    the cap applies only to the number of examples exported for reading.
  * Gradient cosines use the checkpoint saved at the end of each phase, with
    batches drawn using that phase's composition, averaged over many batches.
  * Attention-pooling weights are shown as an illustration only.
"""
from __future__ import annotations

import os
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from .config import LABELS, Config
from .evaluate import predict
from .model import EN, VI, XAlignFact
from .utils import save_json


def load_model(cfg: Config, ckpt_path: str, vocab_size: int, device) -> XAlignFact:
    m = XAlignFact(cfg, vocab_size=vocab_size).to(device)
    m.load_state_dict(torch.load(ckpt_path, map_location=device))
    m.eval()
    return m


# ---------------------------------------------------------------------------
@torch.no_grad()
def tsne_plot(model, en_test: Dict, vi_test: Dict, out_png: str, device, n: int = 1500,
              seed: int = 0) -> None:
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    g = np.random.default_rng(seed)
    zs, ys, ls = [], [], []
    for data, lang in ((en_test, EN), (vi_test, VI)):
        idx = g.choice(len(data["labels"]), size=min(n, len(data["labels"])), replace=False)
        sub = {k: v[torch.as_tensor(idx)] for k, v in data.items()}
        for i in range(0, len(idx), 128):
            out = model(sub["input_ids"][i:i + 128].to(device).long(),
                        sub["attention_mask"][i:i + 128].to(device).long(), lang)
            zs.append(out["z"].float().cpu().numpy())
        ys.append(sub["labels"].numpy())
        ls += [lang] * len(idx)
    Z, Y, L = np.concatenate(zs), np.concatenate(ys), np.array(ls)
    P = model.prototypes.protos.float().cpu().numpy().reshape(-1, Z.shape[1])  # (2*C, d)
    emb = TSNE(n_components=2, perplexity=30, init="pca", random_state=seed).fit_transform(
        np.concatenate([Z, P]))
    E, EP = emb[: len(Z)], emb[len(Z):]
    colors = ["#2e7d32", "#c62828", "#1565c0"]
    fig, ax = plt.subplots(figsize=(7, 6))
    for lang, mk in ((EN, "o"), (VI, "^")):
        for c in range(len(LABELS)):
            m = (L == lang) & (Y == c)
            ax.scatter(E[m, 0], E[m, 1], s=6, alpha=0.3, marker=mk, c=colors[c])
    C = len(LABELS)
    for c in range(C):
        ax.scatter(*EP[c], marker="*", s=300, c=colors[c], edgecolors="k", label=f"EN proto {LABELS[c]}")
        ax.scatter(*EP[C + c], marker="P", s=220, c=colors[c], edgecolors="k", label=f"VI proto {LABELS[c]}")
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(fontsize=7, loc="best")
    ax.set_title(f"t-SNE of z on test sets ({n} EN, {n} VI); prototypes in the same fit")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
def error_analysis(preds_npz: str, split_jsonl: str, out_dir: str, n_examples: int = 25) -> Dict:
    p = np.load(preds_npz)
    df = pd.read_json(split_jsonl, lines=True)
    y, yh, prob = p["labels"], p["preds"], p["probs"]
    k = len(LABELS)
    cm = np.zeros((k, k), dtype=int)
    for a, b in zip(y, yh):
        cm[a, b] += 1
    report = {"confusion_counts": cm.tolist(), "pairs": {}}
    for t in range(k):
        for q in range(k):
            if t == q:
                continue
            idx = np.where((y == t) & (yh == q))[0]
            report["pairs"][f"{LABELS[t]}->{LABELS[q]}"] = int(len(idx))
            ex = df.iloc[idx[:n_examples]].copy()
            ex["confidence"] = prob[idx[:n_examples]].max(1)
            ex.to_csv(f"{out_dir}/errors_{t}_to_{q}.csv", index=False)
    save_json(report, f"{out_dir}/error_counts.json")
    return report


# ---------------------------------------------------------------------------
def gradient_cosines(cfg: Config, ckpt_paths: Dict[str, str], data: Dict, vocab_size: int,
                     device, n_batches: int = 20, seed: int = 0) -> Dict:
    """Cosine similarity between gradients of loss terms on encoder parameters."""
    from .sampler import CurriculumSampler
    from .train import build_batch

    results = {}
    phase_epoch = {ph["name"]: ph["end_epoch"] for ph in cfg.phases}
    for phase, path in ckpt_paths.items():
        model = load_model(cfg, path, vocab_size, device)
        model.train()
        sizes = {"en": len(data["en_train"]["labels"]), "vi": len(data["vi_train"]["labels"]),
                 "par": len(data["par_train"]["labels"])}
        sampler = CurriculumSampler(cfg, sizes, seed)
        params = [p for p in model.encoder.parameters() if p.requires_grad]
        acc: Dict[str, List[float]] = {}
        for _ in range(n_batches):
            batch = build_batch(data, sampler.next(phase_epoch[phase]), device)
            terms = _separate_terms(model, batch)
            grads = {}
            for name, loss in terms.items():
                g = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
                grads[name] = torch.cat([(x if x is not None else torch.zeros_like(p)).flatten()
                                         for x, p in zip(g, params)])
            names = list(grads)
            for i, a in enumerate(names):
                for b in names[i + 1:]:
                    c = torch.nn.functional.cosine_similarity(grads[a], grads[b], dim=0).item()
                    acc.setdefault(f"{a}|{b}", []).append(c)
        results[phase] = {k: {"mean": float(np.mean(v)), "sd": float(np.std(v))} for k, v in acc.items()}
        del model
    return results


def _separate_terms(model, batch) -> Dict[str, torch.Tensor]:
    import torch.nn.functional as F

    from .losses import consistency_loss, prototype_loss, representation_loss

    cfg = model.cfg
    out = {}
    z_en = model.encoder(batch["en_ids"].long(), batch["en_mask"].long())[0]
    z_vi = model.encoder(batch["vi_ids"].long(), batch["vi_mask"].long())[0]
    out["L_EN"] = F.cross_entropy(model.head(EN)(z_en).float(), batch["en_y"])
    out["L_VI"] = F.cross_entropy(model.head(VI)(z_vi).float(), batch["vi_y"])
    if "par_en_ids" in batch:
        pe = model.encoder(batch["par_en_ids"].long(), batch["par_en_mask"].long())[0]
        pv = model.encoder(batch["par_vi_ids"].long(), batch["par_vi_mask"].long())[0]
        out["L_cons"] = consistency_loss(model.head(EN)(pe), model.head(VI)(pv))
        out["L_repr"] = representation_loss(pe, pv)
    if model.prototypes.ready(EN) and model.prototypes.ready(VI):
        out["L_proto"] = 0.5 * (prototype_loss(z_en, batch["en_y"], model.prototypes.protos[EN], cfg.temperature_proto)
                                + prototype_loss(z_vi, batch["vi_y"], model.prototypes.protos[VI], cfg.temperature_proto))
    return out


# ---------------------------------------------------------------------------
def export_translation_sheet(cfg: Config, n: int = 200, seed: int = 0) -> str:
    """EXP-4: stratified sample of CLCS-test pairs for bilingual annotation."""
    df = pd.read_json(f"{cfg.data_dir}/splits/par_test.jsonl", lines=True)
    df["pair_index"] = np.arange(len(df))
    per = n // len(LABELS)
    sample = pd.concat([g.sample(n=min(per, len(g)), random_state=seed)
                        for _, g in df.groupby("label")]).sample(frac=1, random_state=seed)
    sample["label_name"] = sample["label"].map(dict(enumerate(LABELS)))
    sample = sample[["pair_index", "label_name", "claim_en", "claim_vi", "evidence_en", "evidence_vi"]]
    for col in ("annotator", "label_preserved(yes/no)", "error_type(negation/entity/number/other/none)", "note"):
        sample[col] = ""
    path = f"{cfg.data_dir}/translation_annotation_sheet.csv"
    sample.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def clcs_by_annotation(cfg: Config, annotated_csv: str, variant: str, seeds: List[int]) -> pd.DataFrame:
    """CLCS / joint accuracy on pairs judged label-preserving vs. not, per seed."""
    ann = pd.read_csv(annotated_csv)
    ok = ann.set_index("pair_index")["label_preserved(yes/no)"].str.lower().str.startswith("y")
    rows = []
    for s in seeds:
        p = np.load(f"{cfg.root}/runs/{variant}/seed{s}/preds_par_test.npz")
        for flag in (True, False):
            idx = ok[ok == flag].index.values
            if len(idx) == 0:
                continue
            e, v, y = p["en_preds"][idx], p["vi_preds"][idx], p["labels"][idx]
            rows.append({"seed": s, "label_preserved": flag, "n": len(idx),
                         "clcs": float((e == v).mean()), "joint_acc": float(((e == y) & (v == y)).mean())})
    return pd.DataFrame(rows)


@torch.no_grad()
def attention_figure(model, data: Dict, indices: List[int], tokenizer, lang: int, out_png: str,
                     device, max_tokens: int = 60) -> None:
    """Illustration only: attention-pooling weights are not a faithful explanation."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(indices), 1, figsize=(14, 3 * len(indices)))
    axes = np.atleast_1d(axes)
    for ax, i in zip(axes, indices):
        ids = data["input_ids"][i:i + 1].to(device).long()
        mask = data["attention_mask"][i:i + 1].to(device).long()
        out = model(ids, mask, lang)
        L = min(int(mask.sum()), max_tokens)
        toks = [t.replace("▁", "") for t in tokenizer.convert_ids_to_tokens(ids[0, :L].tolist())]
        a = out["alpha"][0, :L].float().cpu().numpy()
        ax.bar(range(L), a)
        ax.set_xticks(range(L)); ax.set_xticklabels(toks, rotation=90, fontsize=6)
        ax.set_title(f"true={LABELS[int(data['labels'][i])]} pred={LABELS[int(out['logits'].argmax())]}",
                     fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
