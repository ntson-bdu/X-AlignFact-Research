"""Part 6 - Retrieved-evidence setting for ViFactCheck (EXP-5, answers AE3 / R1.3).

The headline results use gold evidence. As a supplementary setting, evidence is
replaced by the top-k sentences retrieved with BM25 from the article context
shipped with ViFactCheck (`Context`). No retriever is trained; this measures how
much the verifier depends on gold evidence.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from difflib import SequenceMatcher
from typing import List

from .utils import normalize_text

_SENT_RE = re.compile(r"(?<=[\.\!\?…])\s+|\n+")


def split_sentences(text: str) -> List[str]:
    sents = [s.strip() for s in _SENT_RE.split(str(text)) if s and s.strip()]
    return [s for s in sents if len(s.split()) >= 3] or ([str(text).strip()] if str(text).strip() else [])


def _tok(s: str) -> List[str]:
    return normalize_text(s).split()   # Vietnamese syllables


def bm25_scores(query: str, docs: List[str], k1: float = 1.5, b: float = 0.75) -> List[float]:
    toks = [_tok(d) for d in docs]
    n = len(toks)
    if n == 0:
        return []
    avgdl = sum(len(t) for t in toks) / n or 1.0
    df = Counter(w for t in toks for w in set(t))
    q = _tok(query)
    scores = []
    for t in toks:
        tf = Counter(t)
        s = 0.0
        for w in q:
            if w not in tf:
                continue
            idf = math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5))
            s += idf * tf[w] * (k1 + 1) / (tf[w] + k1 * (1 - b + b * len(t) / avgdl))
        scores.append(s)
    return scores


def bm25_retrieve(claim: str, context: str, k: int) -> List[str]:
    sents = split_sentences(context)
    if not sents:
        return [""]
    sc = bm25_scores(claim, sents)
    order = sorted(range(len(sents)), key=lambda i: -sc[i])[:k]
    return [sents[i] for i in sorted(order)]   # keep document order


def _match(a: str, b: str, thr: float) -> bool:
    a, b = normalize_text(a), normalize_text(b)
    return bool(a) and bool(b) and (a in b or b in a or SequenceMatcher(None, a, b).ratio() >= thr)


def evidence_recall(retrieved: List[List[str]], gold: List[str], thr: float = 0.8) -> float:
    """Mean fraction of gold-evidence sentences matched by a retrieved sentence."""
    vals = []
    for ret, g in zip(retrieved, gold):
        gs = split_sentences(g) or [g]
        hit = sum(any(_match(x, r, thr) for r in ret) for x in gs)
        vals.append(hit / len(gs))
    return float(sum(vals) / max(len(vals), 1))
