"""EXP-4 (AE4b, R1.4) without human annotators: automatic translation-quality analysis
of the CLCS test pairs, and CLCS / joint accuracy stratified by translation quality.

Signals per pair (all reference-free or round-trip, no human labels):
  neg_mismatch    negation present in the EN claim but not in the VI claim, or vice versa
                  (the error type most likely to flip SUPPORTS <-> REFUTES)
  num_mismatch    the sets of numbers in the EN and VI claims differ
  labse_claim     LaBSE cosine similarity EN claim vs VI claim         (--labse)
  chrf_roundtrip  chrF of VI claim translated back to EN vs EN claim   (--roundtrip)

python scripts/translation_quality.py --root $ROOT --labse --roundtrip \
    --variants full mt_xlmr_vi translate_train
Outputs: data/translation_quality.csv, results/clcs_by_translation_quality.csv
If human judgements exist (translation_annotation_sheet.csv filled), use
xalignfact.analysis.clcs_by_annotation instead / in addition.
"""
import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xalignfact.config import DEFAULT_SEEDS  # noqa: E402

NEG_EN = re.compile(r"\b(not|no|never|none|nobody|nothing|neither|nor|without|cannot)\b|n't\b", re.I)
NEG_VI = re.compile(r"\b(không|chẳng|chả|chưa|chớ|đừng|không hề|chưa từng|chẳng hề|đâu có)\b", re.I)
NUM = re.compile(r"\d[\d.,]*")


def numbers(text: str) -> frozenset:
    out = set()
    for m in NUM.findall(str(text)):
        digits = re.sub(r"[.,]", "", m)          # 1,000 (EN) == 1.000 (VI)
        if digits:
            out.add(digits.lstrip("0") or "0")
    return frozenset(out)


def chrf(hyp: str, ref: str, n: int = 6, beta: float = 2.0) -> float:
    """Character n-gram F-score (chrF, Popovic 2015), whitespace removed."""
    h, r = re.sub(r"\s+", "", str(hyp).lower()), re.sub(r"\s+", "", str(ref).lower())
    precs, recs = [], []
    for k in range(1, n + 1):
        hg = [h[i:i + k] for i in range(len(h) - k + 1)]
        rg = [r[i:i + k] for i in range(len(r) - k + 1)]
        if not hg or not rg:
            continue
        from collections import Counter
        ch, cr = Counter(hg), Counter(rg)
        overlap = sum((ch & cr).values())
        precs.append(overlap / len(hg))
        recs.append(overlap / len(rg))
    if not precs:
        return 0.0
    p, rr = np.mean(precs), np.mean(recs)
    return 0.0 if p + rr == 0 else float((1 + beta ** 2) * p * rr / (beta ** 2 * p + rr))


def compute_signals(df: pd.DataFrame, use_labse: bool, use_roundtrip: bool, cache: str) -> pd.DataFrame:
    q = pd.DataFrame({"pair_index": np.arange(len(df)), "label": df["label"].values})
    q["neg_en"] = df["claim_en"].map(lambda s: bool(NEG_EN.search(str(s))))
    q["neg_vi"] = df["claim_vi"].map(lambda s: bool(NEG_VI.search(str(s))))
    q["neg_mismatch"] = q["neg_en"] != q["neg_vi"]
    q["num_mismatch"] = [numbers(a) != numbers(b) for a, b in zip(df["claim_en"], df["claim_vi"])]
    if use_labse:
        from sentence_transformers import SentenceTransformer

        m = SentenceTransformer("sentence-transformers/LaBSE")
        e = m.encode(df["claim_en"].tolist(), normalize_embeddings=True, batch_size=64)
        v = m.encode(df["claim_vi"].tolist(), normalize_embeddings=True, batch_size=64)
        q["labse_claim"] = (e * v).sum(1)
    if use_roundtrip:
        rt_path = f"{cache}/roundtrip_vi_en.csv"
        if os.path.exists(rt_path):
            rt = pd.read_csv(rt_path)["claim_vi_to_en"].tolist()
        else:
            from deep_translator import GoogleTranslator

            tr = GoogleTranslator(source="vi", target="en")
            rt = []
            for s in df["claim_vi"]:
                try:
                    rt.append(tr.translate(str(s)))
                except Exception:
                    rt.append("")
            pd.DataFrame({"claim_vi_to_en": rt}).to_csv(rt_path, index=False)
        n_fail = sum(1 for h in rt if not str(h).strip() or str(h) == "nan")
        print(f"round-trip: {n_fail}/{len(rt)} translations empty/failed")
        if n_fail > 0.1 * len(rt):
            print("WARNING: >10% round-trip translations failed; chrF is NOT reported. "
                  f"Delete {rt_path} and retry later (Google Translate rate limit).")
            os.remove(rt_path)
        else:
            q["chrf_roundtrip"] = [chrf(h, r) if str(h).strip() else np.nan
                                   for h, r in zip(rt, df["claim_en"])]
    return q


def stratify(root: str, q: pd.DataFrame, variants, seeds) -> pd.DataFrame:
    groups = {"all": np.ones(len(q), bool),
              "negation_mismatch": q["neg_mismatch"].values,
              "negation_match": ~q["neg_mismatch"].values,
              "number_mismatch": q["num_mismatch"].values,
              "no_flag": ~(q["neg_mismatch"] | q["num_mismatch"]).values}
    for col in ("labse_claim", "chrf_roundtrip"):
        if col in q:
            qs = pd.qcut(q[col].rank(method="first"), 4, labels=["Q1(low)", "Q2", "Q3", "Q4(high)"])
            for lab in qs.cat.categories:
                groups[f"{col}_{lab}"] = (qs == lab).values
    rows = []
    for v in variants:
        for s in seeds:
            p = f"{root}/runs/{v}/seed{s}/preds_par_test.npz"
            if not os.path.exists(p):
                continue
            d = np.load(p)
            e, vi, y = d["en_preds"], d["vi_preds"], d["labels"]
            for g, m in groups.items():
                if m.sum() == 0:
                    continue
                rows.append({"variant": v, "seed": s, "group": g, "n": int(m.sum()),
                             "clcs": float((e[m] == vi[m]).mean()),
                             "joint_acc": float(((e[m] == y[m]) & (vi[m] == y[m])).mean()),
                             "vi_acc": float((vi[m] == y[m]).mean())})
    raw = pd.DataFrame(rows)
    return (raw.groupby(["variant", "group", "n"])[["clcs", "joint_acc", "vi_acc"]]
               .agg(["mean", "std"]).round(4).reset_index())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--variants", nargs="*", default=["full", "mt_xlmr_vi", "translate_train"])
    ap.add_argument("--seeds", nargs="*", type=int, default=DEFAULT_SEEDS)
    ap.add_argument("--labse", action="store_true")
    ap.add_argument("--roundtrip", action="store_true")
    a = ap.parse_args()

    df = pd.read_json(f"{a.root}/data/splits/par_test.jsonl", lines=True)
    q = compute_signals(df, a.labse, a.roundtrip, f"{a.root}/data")
    q.to_csv(f"{a.root}/data/translation_quality.csv", index=False)
    # Pairs flagged by the heuristics, for a quick manual check by the authors
    flag = q["neg_mismatch"] | q["num_mismatch"]
    sheet = df.loc[flag.values, ["claim_en", "claim_vi", "label"]].copy()
    sheet.insert(0, "pair_index", np.where(flag.values)[0])
    sheet["flag"] = np.where(q.loc[flag, "neg_mismatch"], "negation", "number")
    sheet["real_error(yes/no)"] = ""
    sheet["label_preserved(yes/no)"] = ""
    sheet.to_csv(f"{a.root}/data/flagged_translation_pairs.csv", index=False, encoding="utf-8-sig")
    print(f"flagged pairs for manual check: {len(sheet)} -> data/flagged_translation_pairs.csv")
    print(f"pairs: {len(q)} | negation mismatch: {q['neg_mismatch'].mean():.1%} "
          f"| number mismatch: {q['num_mismatch'].mean():.1%}")
    for col in ("labse_claim", "chrf_roundtrip"):
        if col in q:
            print(f"{col}: mean={q[col].mean():.3f} median={q[col].median():.3f}")
    out = stratify(a.root, q, a.variants, a.seeds)
    os.makedirs(f"{a.root}/results", exist_ok=True)
    out.to_csv(f"{a.root}/results/clcs_by_translation_quality.csv", index=False)
    print(out.to_string(index=False))
