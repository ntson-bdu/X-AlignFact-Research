"""EXP-10 - Explainability analyses that can be checked quantitatively.

1. Evidence reliance (claim-only / evidence-only probes): how much accuracy and cross-lingual
   consistency survive when the evidence (or the claim) is removed. A verifier that keeps its
   accuracy without evidence relies on claim artefacts rather than on verification.
2. Token attributions with Integrated Gradients (Sundararajan et al., 2017) on the word
   embeddings, w.r.t. the predicted class.
3. Faithfulness of the attributions (ERASER comprehensiveness / sufficiency, DeYoung et al., 2020),
   compared with the pooling-attention weights and a random ranking. This tests, rather than
   assumes, whether attention weights explain the decision.
4. Cross-lingual attribution consistency on parallel pairs: does the model rely on the evidence
   (vs. the claim) and on numbers to the same extent in English and in Vietnamese?

All functions work on the tensors produced by data.tokenize_all (`<s> [LANG] claim </s></s> evidence </s>`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch


@dataclass
class SpecialIds:
    cls: int
    sep: int
    pad: int
    mask: int

    @classmethod
    def from_tokenizer(cls, tok) -> "SpecialIds":
        return cls(tok.cls_token_id, tok.sep_token_id, tok.pad_token_id, tok.mask_token_id)


# ---------------------------------------------------------------------------
# segments
def segments(ids: torch.Tensor, mask: torch.Tensor, sp: SpecialIds) -> Dict[str, List[int]]:
    """Positions of claim and evidence tokens in one sequence (1-D tensors)."""
    n = int(mask.sum())
    row = ids[:n].tolist()
    seps = [i for i, t in enumerate(row) if t == sp.sep]
    if not seps:
        return {"claim": list(range(2, n)), "evidence": [], "first_sep": n, "last": n - 1}
    s1 = seps[0]
    last = n - 1 if row[n - 1] == sp.sep else n
    ev_start = s1 + 2 if len(seps) > 1 and seps[1] == s1 + 1 else s1 + 1
    return {"claim": list(range(2, s1)), "evidence": list(range(ev_start, last)),
            "first_sep": s1, "last": last}


def _pack(rows: List[List[int]], length: int, pad: int):
    ids = torch.full((len(rows), length), pad, dtype=torch.long)
    mask = torch.zeros((len(rows), length), dtype=torch.long)
    for i, r in enumerate(rows):
        r = r[:length]
        ids[i, :len(r)] = torch.tensor(r, dtype=torch.long)
        mask[i, :len(r)] = 1
    return ids, mask


def ablate_segment(data: Dict, sp: SpecialIds, keep: str, prefix: str = "") -> Dict:
    """keep='claim' -> `<s> [LANG] claim </s>`; keep='evidence' -> `<s> [LANG] </s></s> evidence </s>`."""
    ids_all, mask_all = data[f"{prefix}input_ids"], data[f"{prefix}attention_mask"]
    rows = []
    for ids, mask in zip(ids_all, mask_all):
        seg = segments(ids, mask, sp)
        row = ids[: int(mask.sum())].tolist()
        if keep == "claim":
            rows.append(row[: seg["first_sep"]] + [sp.sep])
        else:
            rows.append(row[:2] + [sp.sep, sp.sep] + [row[i] for i in seg["evidence"]] + [sp.sep])
    ids, mask = _pack(rows, ids_all.shape[1], sp.pad)
    out = dict(data)
    out[f"{prefix}input_ids"], out[f"{prefix}attention_mask"] = ids, mask
    return out


# ---------------------------------------------------------------------------
# attributions
def _embeddings(model):
    return model.encoder.encoder.get_input_embeddings()


def integrated_gradients(model, ids: torch.Tensor, mask: torch.Tensor, lang: int, target: int,
                         sp: SpecialIds, steps: int = 32) -> np.ndarray:
    """IG for one sequence; baseline = <pad> embedding on content tokens, specials kept."""
    device = next(model.parameters()).device
    ids, mask = ids.to(device).long().unsqueeze(0), mask.to(device).long().unsqueeze(0)
    emb = _embeddings(model)
    with torch.no_grad():
        x = emb(ids)
        base_ids = ids.clone()
        special = (ids == sp.cls) | (ids == sp.sep) | (torch.arange(ids.shape[1], device=device) == 1)
        base_ids[~special & (mask == 1)] = sp.pad
        b = emb(base_ids)
    alphas = torch.linspace(1.0 / steps, 1.0, steps, device=device).view(-1, 1, 1)
    path = (b + alphas * (x - b)).detach().requires_grad_(True)       # [steps, T, d]
    out = model(None, mask.expand(steps, -1), lang, inputs_embeds=path)
    prob = torch.softmax(out["logits"].float(), -1)[:, target].sum()
    (grad,) = torch.autograd.grad(prob, path)
    attr = ((x - b) * grad.mean(0, keepdim=True)).sum(-1).squeeze(0)  # [T]
    attr = attr * mask.squeeze(0)
    return attr.detach().float().cpu().numpy()


@torch.no_grad()
def _probs(model, ids, mask, lang, bs=64):
    device = next(model.parameters()).device
    out = []
    for i in range(0, len(ids), bs):
        lo = model(ids[i:i + bs].to(device), mask[i:i + bs].to(device), lang)["logits"]
        out.append(torch.softmax(lo.float(), -1).cpu())
    return torch.cat(out)


def faithfulness(model, ids: torch.Tensor, mask: torch.Tensor, lang: int, scores: Dict[str, np.ndarray],
                 sp: SpecialIds, ks=(0.05, 0.10, 0.20, 0.50), rng: Optional[np.random.Generator] = None
                 ) -> Dict[str, Dict[str, float]]:
    """ERASER comprehensiveness (drop top-k) and sufficiency (keep only top-k), AOPC over ks.
    Tokens are replaced by <mask> so that sequence length and positions are unchanged."""
    rng = rng or np.random.default_rng(0)
    seg = segments(ids, mask, sp)
    content = np.array(seg["claim"] + seg["evidence"])
    if len(content) == 0:
        return {}
    scores = dict(scores)
    scores["random"] = rng.random(ids.shape[0])
    rows, keys = [ids.clone()], [("orig", None, None)]
    for name, s in scores.items():
        order = content[np.argsort(-s[content], kind="stable")]
        for k in ks:
            top = order[: max(1, math.ceil(k * len(content)))]
            drop = ids.clone()
            drop[torch.as_tensor(top)] = sp.mask
            keep = ids.clone()
            rest = np.setdiff1d(content, top)
            if len(rest):
                keep[torch.as_tensor(rest)] = sp.mask
            rows += [drop, keep]
            keys += [(name, k, "comp"), (name, k, "suff")]
    batch = torch.stack(rows).long()
    p = _probs(model, batch, mask.unsqueeze(0).expand(len(rows), -1).long(), lang)
    yhat = int(p[0].argmax())
    p0 = float(p[0, yhat])
    res: Dict[str, Dict[str, List[float]]] = {}
    for (name, k, kind), pi in zip(keys[1:], p[1:]):
        res.setdefault(name, {"comp": [], "suff": []})[kind].append(p0 - float(pi[yhat]))
    return {n: {"comprehensiveness": float(np.mean(v["comp"])), "sufficiency": float(np.mean(v["suff"]))}
            for n, v in res.items()}


def attribution_profile(attr: np.ndarray, ids: torch.Tensor, mask: torch.Tensor, sp: SpecialIds,
                        digit_positions: Optional[set] = None) -> Dict[str, float]:
    seg = segments(ids, mask, sp)
    a = np.abs(attr)
    tot = a[seg["claim"] + seg["evidence"]].sum() + 1e-12
    out = {"evidence_share": float(a[seg["evidence"]].sum() / tot)}
    if digit_positions is not None:
        out["digit_share"] = float(a[sorted(digit_positions)].sum() / tot) if digit_positions else float("nan")
    return out


def digit_positions(ids: torch.Tensor, mask: torch.Tensor, tok) -> set:
    toks = tok.convert_ids_to_tokens(ids[: int(mask.sum())].tolist())
    return {i for i, t in enumerate(toks) if any(ch.isdigit() for ch in t)}


# ---------------------------------------------------------------------------
# drivers
def explain_single(model, data: Dict, lang: int, sp: SpecialIds, n: int, steps: int = 32,
                   seed: int = 0) -> Dict:
    """IG + attention faithfulness on n random examples of a monolingual test set."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(data["labels"]), size=min(n, len(data["labels"])), replace=False)
    device = next(model.parameters()).device
    rows = {"ig": [], "attention": [], "random": [], "evidence_share": []}
    for i in idx:
        ids, mask = data["input_ids"][i].long(), data["attention_mask"][i].long()
        with torch.no_grad():
            out = model(ids.unsqueeze(0).to(device), mask.unsqueeze(0).to(device), lang)
        yhat = int(out["logits"].argmax(-1))
        alpha = out["alpha"].squeeze(0).float().cpu().numpy()
        ig = integrated_gradients(model, ids, mask, lang, yhat, sp, steps)
        f = faithfulness(model, ids, mask, lang, {"ig": ig, "attention": alpha}, sp, rng=rng)
        for name in ("ig", "attention", "random"):
            if name in f:
                rows[name].append(f[name])
        rows["evidence_share"].append(attribution_profile(ig, ids, mask, sp)["evidence_share"])
    summ = {}
    for name in ("ig", "attention", "random"):
        if rows[name]:
            summ[name] = {k: float(np.mean([r[k] for r in rows[name]])) for k in rows[name][0]}
    summ["ig_evidence_share"] = float(np.mean(rows["evidence_share"]))
    summ["n"] = int(len(idx))
    return summ


def explain_parallel(model, data: Dict, sp: SpecialIds, tok, n: int, steps: int = 32, seed: int = 0) -> Dict:
    """Cross-lingual attribution consistency on n parallel pairs (EN head on EN, VI head on VI)."""
    from scipy.stats import spearmanr

    from .model import EN, VI

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(data["labels"]), size=min(n, len(data["labels"])), replace=False)
    device = next(model.parameters()).device
    prof = {"en": [], "vi": []}
    agree = []
    for i in idx:
        preds = {}
        for side, lang in (("en", EN), ("vi", VI)):
            ids = data[f"{side}_input_ids"][i].long()
            mask = data[f"{side}_attention_mask"][i].long()
            with torch.no_grad():
                yhat = int(model(ids.unsqueeze(0).to(device), mask.unsqueeze(0).to(device), lang)["logits"].argmax(-1))
            preds[side] = yhat
            ig = integrated_gradients(model, ids, mask, lang, yhat, sp, steps)
            prof[side].append(attribution_profile(ig, ids, mask, sp, digit_positions(ids, mask, tok)))
        agree.append(preds["en"] == preds["vi"])
    ev_en = np.array([p["evidence_share"] for p in prof["en"]])
    ev_vi = np.array([p["evidence_share"] for p in prof["vi"]])
    dg_en = np.array([p["digit_share"] for p in prof["en"]])
    dg_vi = np.array([p["digit_share"] for p in prof["vi"]])
    both = ~np.isnan(dg_en) & ~np.isnan(dg_vi)
    agree = np.array(agree)
    return {
        "n": int(len(idx)),
        "evidence_share_en": float(ev_en.mean()), "evidence_share_vi": float(ev_vi.mean()),
        "evidence_share_spearman": float(spearmanr(ev_en, ev_vi).correlation),
        "evidence_share_absdiff_agree": float(np.abs(ev_en - ev_vi)[agree].mean()) if agree.any() else float("nan"),
        "evidence_share_absdiff_disagree": float(np.abs(ev_en - ev_vi)[~agree].mean()) if (~agree).any() else float("nan"),
        "n_with_digits": int(both.sum()),
        "digit_share_en": float(dg_en[both].mean()) if both.any() else float("nan"),
        "digit_share_vi": float(dg_vi[both].mean()) if both.any() else float("nan"),
        "digit_share_spearman": float(spearmanr(dg_en[both], dg_vi[both]).correlation) if both.sum() > 2 else float("nan"),
    }
