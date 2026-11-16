"""Part 4 - Evaluation.

CLCS family (computed on held-out parallel pairs, EN head on EN side, VI head on VI side):
  CLCS       fraction of pairs with identical predicted labels
  CLCS-p     agreement among pairs where at least one head is correct
  kappa      Cohen's kappa between EN and VI predictions (agreement beyond chance)
  joint_acc  fraction of pairs where both heads are correct
CLCS alone is maximised by a constant predictor, so it is always reported together
with joint accuracy.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score, confusion_matrix, precision_recall_fscore_support

from .config import LABELS


@torch.no_grad()
def predict(model, data: Dict, head_lang: int, device, batch_size: int = 128) -> Dict[str, np.ndarray]:
    preds, probs = [], []
    n = len(data["labels"])
    for i in range(0, n, batch_size):
        out = model(data["input_ids"][i:i + batch_size].to(device).long(),
                    data["attention_mask"][i:i + batch_size].to(device).long(), head_lang)
        p = torch.softmax(out["logits"].float(), -1).cpu()
        probs.append(p)
        preds.append(p.argmax(-1))
    return {"labels": data["labels"].numpy(), "preds": torch.cat(preds).numpy(),
            "probs": torch.cat(probs).numpy()}


@torch.no_grad()
def predict_parallel(model, data: Dict, device, batch_size: int = 128) -> Dict[str, np.ndarray]:
    from .model import EN, VI

    en = predict(model, {"input_ids": data["en_input_ids"], "attention_mask": data["en_attention_mask"],
                         "labels": data["labels"]}, EN, device, batch_size)
    vi = predict(model, {"input_ids": data["vi_input_ids"], "attention_mask": data["vi_attention_mask"],
                         "labels": data["labels"]}, VI, device, batch_size)
    return {"labels": en["labels"], "en_preds": en["preds"], "vi_preds": vi["preds"],
            "en_probs": en["probs"], "vi_probs": vi["probs"]}


def classification_metrics(labels: np.ndarray, preds: np.ndarray) -> Dict:
    k = list(range(len(LABELS)))
    p, r, f, s = precision_recall_fscore_support(labels, preds, labels=k, zero_division=0)
    return {
        "macro_f1": float(np.mean(f)),
        "accuracy": float((labels == preds).mean()),
        "per_class": {LABELS[i]: {"precision": float(p[i]), "recall": float(r[i]),
                                  "f1": float(f[i]), "support": int(s[i])} for i in k},
        "confusion_counts": confusion_matrix(labels, preds, labels=k).tolist(),  # rows = true
        "n": int(len(labels)),
    }


def clcs_metrics(pp: Dict[str, np.ndarray]) -> Dict:
    y, e, v = pp["labels"], pp["en_preds"], pp["vi_preds"]
    agree = e == v
    one_ok = (e == y) | (v == y)
    both_ok = (e == y) & (v == y)
    per_class = {LABELS[c]: float(agree[y == c].mean()) for c in range(len(LABELS)) if (y == c).any()}
    return {
        "n_pairs": int(len(y)),
        "clcs": float(agree.mean()),
        "clcs_p": float(agree[one_ok].mean()) if one_ok.any() else float("nan"),
        "kappa": float(cohen_kappa_score(e, v)),
        "joint_accuracy": float(both_ok.mean()),
        "en_macro_f1": classification_metrics(y, e)["macro_f1"],
        "vi_macro_f1": classification_metrics(y, v)["macro_f1"],
        "per_class_clcs": per_class,
    }
