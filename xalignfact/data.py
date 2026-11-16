"""Part 1 - Leakage-free data preparation.

Fixes vs. v1:
  * English test = official nli_fever *test* split (v1 used 5% of the training pool).
  * English dev  = stratified subsample of the official nli_fever *dev* split.
  * Vietnamese dev (723) is used for model selection; test (1,447) only for reporting.
  * Parallel pairs are claim-disjoint from every English split and are split into
    par_train (alignment losses), par_dev (CLCS for model selection) and
    par_test (CLCS reporting).
  * Hard negatives are not used.
  * Every pairwise overlap between splits is measured by normalised-claim hash
    and written to data/overlap_report.json; train/test overlap must be 0.

Outputs (under cfg.data_dir):
  splits/*.jsonl            human-readable rows with claim hashes
  tensors/*.pt              tokenised tensors used by training/evaluation
  manifest.json             counts, label distributions, decisions, file hashes
  overlap_report.json       pairwise claim overlaps
"""
from __future__ import annotations

import os
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from .config import LABEL2ID, LABELS, Config
from .utils import Logger, file_sha256, load_json, save_json, text_hash

LANG_TOKENS = ["[EN]", "[VI]"]


# ---------------------------------------------------------------------------
# Loading raw datasets
# ---------------------------------------------------------------------------
def _detect_claim_field(df: pd.DataFrame, log: Logger) -> str:
    """FEVER claims are short sentences; evidence passages are longer."""
    sample = df.sample(n=min(5000, len(df)), random_state=0)
    med = {c: sample[c].astype(str).str.split().str.len().median() for c in ("premise", "hypothesis")}
    field = min(med, key=med.get)
    log(f"[data] nli_fever median words: {med} -> claim field = '{field}'")
    return field


def load_fever(cfg: Config, log: Logger) -> Tuple[Dict[str, pd.DataFrame], str]:
    from datasets import load_dataset

    ds = load_dataset(cfg.fever_hf)
    log(f"[data] {cfg.fever_hf} splits: { {k: len(v) for k, v in ds.items()} }")
    raw = {k: ds[k].to_pandas() for k in ds.keys()}
    claim_field = cfg.fever_claim_field
    if claim_field == "auto":
        claim_field = _detect_claim_field(raw["train"], log)
    ev_field = "hypothesis" if claim_field == "premise" else "premise"

    out = {}
    for split, df in raw.items():
        lab = df[cfg.fever_label_field].map(LABEL2ID)
        frame = pd.DataFrame({
            "claim": df[claim_field].astype(str),
            "evidence": df[ev_field].astype(str),
            "label": lab,
        })
        frame = frame[frame["label"].notna()]
        frame = frame[frame["evidence"].str.strip().str.len() >= 5]
        frame = frame[frame["claim"].str.strip().str.len() > 0]
        frame["label"] = frame["label"].astype(int)
        frame["claim_hash"] = frame["claim"].map(text_hash)
        out[split] = frame.reset_index(drop=True)
    return out, claim_field


def load_vifactcheck(cfg: Config) -> Dict[str, pd.DataFrame]:
    """ViFactCheck columns: Statement, Evidence, Context, labels.

    [AUTHOR CHECK] label ids are used as-is (0=SUPPORTS, 1=REFUTES, 2=NEI, as in
    v1). Confirm this mapping against the dataset card before the final runs.
    """
    from datasets import load_dataset

    ds = load_dataset(cfg.vifact_hf)
    out = {}
    for split in ("train", "dev", "test"):
        df = ds[split].to_pandas()
        frame = pd.DataFrame({
            "claim": df["Statement"].astype(str),
            "evidence": df["Evidence"].astype(str),
            "context": df["Context"].astype(str) if "Context" in df else "",
            "label": df["labels"].astype(int),
        })
        frame["claim_hash"] = frame["claim"].map(text_hash)
        out[split] = frame.reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Splitting helpers
# ---------------------------------------------------------------------------
def stratified_sample(df: pd.DataFrame, n_per_class: int | None, n_total: int | None,
                      seed: int) -> pd.DataFrame:
    parts = []
    if n_per_class is not None:
        for lbl in sorted(df["label"].unique()):
            sub = df[df["label"] == lbl]
            parts.append(sub.sample(n=min(len(sub), n_per_class), random_state=seed))
        return pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    if n_total is None or n_total >= len(df):
        return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    frac = n_total / len(df)
    for lbl in sorted(df["label"].unique()):
        sub = df[df["label"] == lbl]
        parts.append(sub.sample(n=int(round(len(sub) * frac)), random_state=seed))
    return pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def overlap_report(sets: Dict[str, Iterable[str]]) -> Dict[str, int]:
    hashed = {k: set(v) for k, v in sets.items()}
    names = list(hashed)
    rep = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            rep[f"{a}__{b}"] = len(hashed[a] & hashed[b])
    return rep


def load_legacy_parallel(path: str, claim_field: str, log: Logger) -> pd.DataFrame:
    """Load v1 parallel_en_vi.json. v1 used nli_fever 'premise' as the claim."""
    df = pd.read_json(path)
    if claim_field == "hypothesis":
        log("[data] claim field is 'hypothesis' but v1 used 'premise' as claim: "
            "swapping claim/evidence columns of the legacy parallel corpus.")
        df = df.rename(columns={"claim_en": "evidence_en", "evidence_en": "claim_en",
                                "claim_vi": "evidence_vi", "evidence_vi": "claim_vi"})
    df["label"] = df["label"].astype(int)
    df["claim_hash"] = df["claim_en"].map(text_hash)
    df["evidence_hash"] = df["evidence_en"].map(text_hash)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main entry: build all splits
# ---------------------------------------------------------------------------
def build_splits(cfg: Config, log: Logger | None = None) -> Dict[str, pd.DataFrame]:
    log = log or Logger(f"{cfg.data_dir}/prepare.log")
    os.makedirs(f"{cfg.data_dir}/splits", exist_ok=True)
    seed = cfg.data_seed

    fever, claim_field = load_fever(cfg, log)
    vi = load_vifactcheck(cfg)

    # ---- parallel corpus -------------------------------------------------
    eval_hashes = set(fever["dev"]["claim_hash"]) | set(fever["test"]["claim_hash"])
    if cfg.parallel_legacy_path:
        par = load_legacy_parallel(cfg.parallel_legacy_path, claim_field, log)
        par_source = f"legacy:{os.path.basename(cfg.parallel_legacy_path)}"
        par_source_sha = file_sha256(cfg.parallel_legacy_path)
        n_before = len(par)
        par = par[~par["claim_hash"].isin(eval_hashes)].reset_index(drop=True)
        log(f"[data] legacy parallel: {n_before} pairs, {n_before - len(par)} dropped "
            f"(claim also in FEVER dev/test)")
    else:
        from .translate import translate_pairs

        pool = fever["train"][~fever["train"]["claim_hash"].isin(eval_hashes)]
        pool = pool.drop_duplicates("claim_hash")
        src = stratified_sample(pool, None, cfg.parallel_n_new, seed)
        par = translate_pairs(src, f"{cfg.data_dir}/parallel_en_vi_v2.json", log)
        par["claim_hash"] = par["claim_en"].map(text_hash)
        par["evidence_hash"] = par["evidence_en"].map(text_hash)
        par_source = "google-translate via deep-translator (see translation_meta.json)"
        par_source_sha = file_sha256(f"{cfg.data_dir}/parallel_en_vi_v2.json")

    n_before = len(par)
    par = par.drop_duplicates("claim_hash").reset_index(drop=True)
    if len(par) < n_before:
        log(f"[data] parallel: {n_before - len(par)} duplicate claims dropped "
            f"(a claim must not appear in more than one parallel split)")
    par = par.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    par_test = par.iloc[: cfg.parallel_test_n]
    par_dev = par.iloc[cfg.parallel_test_n: cfg.parallel_test_n + cfg.parallel_dev_n]
    par_train = par.iloc[cfg.parallel_test_n + cfg.parallel_dev_n:]

    # ---- English test source ----------------------------------------------------
    # The FEVER shared-task test labels are hidden, so nli_fever 'test' may carry no
    # gold labels. Standard practice then: the labelled FEVER dev split is the test set,
    # and a claim-disjoint dev set is held out from the training pool.
    log(f"[data] labelled rows per nli_fever split: { {k: len(v) for k, v in fever.items()} }")
    if len(fever["test"]) > 0:
        test_source, en_test = "nli_fever:test", fever["test"]
        dev_from_train = False
    elif len(fever["dev"]) > 0:
        test_source, en_test = "nli_fever:dev (test labels unavailable)", fever["dev"]
        dev_from_train = True
    else:
        raise RuntimeError(f"No labelled FEVER evaluation split ('{cfg.fever_label_field}').")
    log(f"[data] English test source = {test_source}")

    # ---- English train pool: exclude every claim that appears elsewhere ------
    # Claim-level disjointness (standard for FEVER; evidence sentences are shared across claims).
    par_hashes = set(par["claim_hash"])
    tr = fever["train"]
    excluded = tr["claim_hash"].isin(par_hashes | eval_hashes)
    log(f"[data] FEVER train pool: {len(tr)} rows, {int(excluded.sum())} excluded "
        f"(claims shared with parallel corpus or dev/test)")
    pool = tr[~excluded]

    if dev_from_train:
        # hold out whole claims (all their rows) for the English dev set
        # one row per claim, class-balanced (SUPPORTS claims have many evidence rows,
        # so sampling rows and expanding to claims would skew the label distribution)
        uniq = pool.drop_duplicates("claim_hash")
        en_dev = stratified_sample(uniq, cfg.fever_dev_max // cfg.num_labels, None, seed)
        dev_claims = set(en_dev["claim_hash"])
        pool = pool[~pool["claim_hash"].isin(dev_claims)]
        log(f"[data] English dev = {len(en_dev)} rows held out from the train pool "
            f"({len(dev_claims)} claims)")

    avail = pool["label"].value_counts()
    per_class = int(min(cfg.fever_train_per_class, avail.min()))
    if per_class < cfg.fever_train_per_class:
        log(f"[data] smallest class has {int(avail.min())} rows after exclusions -> balanced "
            f"English train uses {per_class} per class (v1: {cfg.fever_train_per_class})")
    en_train = stratified_sample(pool, per_class, None, seed)

    train_like = set(en_train["claim_hash"]) | set(par["claim_hash"])
    n0 = len(en_test)
    en_test = en_test[~en_test["claim_hash"].isin(train_like)].reset_index(drop=True)
    log(f"[data] FEVER test: {n0} rows, {n0 - len(en_test)} dropped (claim in train/parallel)")

    if not dev_from_train:
        dev_pool = fever["dev"]
        n0 = len(dev_pool)
        dev_pool = dev_pool[~dev_pool["claim_hash"].isin(train_like | set(en_test["claim_hash"]))]
        log(f"[data] FEVER dev: {n0} rows, {n0 - len(dev_pool)} dropped (claim in train/parallel/test)")
        en_dev = stratified_sample(dev_pool, None, cfg.fever_dev_max, seed)

    # ---- Vietnamese -------------------------------------------------------
    vi_eval = set(vi["dev"]["claim_hash"]) | set(vi["test"]["claim_hash"])
    n0 = len(vi["train"])
    vi_train = vi["train"][~vi["train"]["claim_hash"].isin(vi_eval)].reset_index(drop=True)
    log(f"[data] ViFactCheck train: {n0} rows, {n0 - len(vi_train)} dropped (statement in dev/test)")
    vi_test = vi["test"]
    n0 = len(vi["dev"])
    vi_dev = vi["dev"][~vi["dev"]["claim_hash"].isin(set(vi_test["claim_hash"]))].reset_index(drop=True)
    log(f"[data] ViFactCheck dev: {n0} rows, {n0 - len(vi_dev)} dropped (statement in test)")

    splits = {
        "en_train": en_train, "en_dev": en_dev, "en_test": en_test,
        "vi_train": vi_train, "vi_dev": vi_dev, "vi_test": vi_test,
        "par_train": par_train, "par_dev": par_dev, "par_test": par_test,
    }

    # ---- overlap report -------------------------------------------------------
    rep = overlap_report({k: v["claim_hash"] for k, v in splits.items()})
    save_json(rep, f"{cfg.data_dir}/overlap_report.json")
    must_be_zero = [k for k in rep if ("test" in k or "dev" in k) and ("train" in k)]
    must_be_zero += ["par_dev__par_test", "en_dev__en_test", "vi_dev__vi_test",
                     "en_test__par_test", "en_test__par_dev"]
    bad = {k: rep[k] for k in must_be_zero if rep.get(k, 0) > 0}
    if bad:
        raise RuntimeError(f"Split overlap detected: {bad}")
    log("[data] overlap check passed (all train/dev/test claim overlaps are 0)")

    for name, df in splits.items():
        df.to_json(f"{cfg.data_dir}/splits/{name}.jsonl", orient="records",
                   lines=True, force_ascii=False)

    manifest = {
        "claim_field": claim_field,
        "english_test_source": test_source,
        "english_dev_source": "held out from nli_fever train (claim-disjoint)" if dev_from_train
                              else "nli_fever:dev (stratified subsample)",
        "en_train_per_class": per_class,
        "parallel_source": par_source,
        "parallel_source_sha256": par_source_sha,
        "data_seed": seed,
        "counts": {k: len(v) for k, v in splits.items()},
        "label_distribution": {
            k: {LABELS[i]: int((v["label"] == i).sum()) for i in range(len(LABELS))}
            for k, v in splits.items()
        },
        "split_files_sha256": {
            k: file_sha256(f"{cfg.data_dir}/splits/{k}.jsonl") for k in splits
        },
    }
    save_json(manifest, f"{cfg.data_dir}/manifest.json")
    log(f"[data] counts: {manifest['counts']}")
    return splits


def load_splits(cfg: Config) -> Dict[str, pd.DataFrame]:
    d = f"{cfg.data_dir}/splits"
    return {f[:-6]: pd.read_json(f"{d}/{f}", lines=True)
            for f in sorted(os.listdir(d)) if f.endswith(".jsonl")}


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------
def get_tokenizer(cfg: Config):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.encoder)
    tok.add_special_tokens({"additional_special_tokens": LANG_TOKENS})
    return tok


def encode_pairs(tok, claims: List[str], evidences: List[str], lang: str,
                 max_length: int, chunk: int = 2048) -> Dict:
    """`<s> [LANG] claim </s></s> evidence </s>`; the longer side is truncated first."""
    import torch

    prefix = "[EN] " if lang == "en" else "[VI] "
    ids, masks = [], []
    for i in range(0, len(claims), chunk):
        enc = tok([prefix + c for c in claims[i:i + chunk]], list(evidences[i:i + chunk]),
                  truncation="longest_first", max_length=max_length,
                  padding="max_length", return_tensors="pt")
        ids.append(enc["input_ids"].to(torch.int32))
        masks.append(enc["attention_mask"].to(torch.int8))
    return {"input_ids": torch.cat(ids), "attention_mask": torch.cat(masks)}


def tokenize_all(cfg: Config, splits: Dict[str, pd.DataFrame] | None = None,
                 log: Logger | None = None) -> None:
    import torch

    log = log or Logger(f"{cfg.data_dir}/prepare.log")
    splits = splits or load_splits(cfg)
    tok = get_tokenizer(cfg)
    out_dir = f"{cfg.data_dir}/tensors"
    os.makedirs(out_dir, exist_ok=True)

    for name, df in splits.items():
        labels = torch.tensor(df["label"].values, dtype=torch.long)
        if name.startswith("par_"):
            en = encode_pairs(tok, df["claim_en"].tolist(), df["evidence_en"].tolist(), "en", cfg.max_length)
            vi = encode_pairs(tok, df["claim_vi"].tolist(), df["evidence_vi"].tolist(), "vi", cfg.max_length)
            data = {"en_input_ids": en["input_ids"], "en_attention_mask": en["attention_mask"],
                    "vi_input_ids": vi["input_ids"], "vi_attention_mask": vi["attention_mask"],
                    "labels": labels}
        else:
            lang = "en" if name.startswith("en_") else "vi"
            enc = encode_pairs(tok, df["claim"].tolist(), df["evidence"].tolist(), lang, cfg.max_length)
            data = {**enc, "labels": labels}
        torch.save(data, f"{out_dir}/{name}.pt")
        trunc = float((data.get("attention_mask", data.get("en_attention_mask")).sum(1)
                       >= cfg.max_length).float().mean())
        log(f"[tok] {name}: {len(labels)} rows, {trunc:.1%} reach max_length={cfg.max_length}")

    meta = load_json(f"{cfg.data_dir}/manifest.json")
    meta["tokenizer"] = cfg.encoder
    meta["vocab_size"] = len(tok)
    meta["max_length"] = cfg.max_length
    save_json(meta, f"{cfg.data_dir}/manifest.json")


def build_retrieved_vi_test(cfg: Config, log: Logger | None = None) -> None:
    """EXP-5: ViFactCheck test with BM25-retrieved evidence from the article context."""
    import torch

    from .retrieval import bm25_retrieve, evidence_recall

    log = log or Logger(f"{cfg.data_dir}/prepare.log")
    df = load_splits(cfg)["vi_test"]
    tok = get_tokenizer(cfg)
    summary = {}
    for k in cfg.bm25_topk:
        retrieved = [bm25_retrieve(c, ctx, k) for c, ctx in zip(df["claim"], df["context"])]
        texts = [" ".join(r) for r in retrieved]
        rec = evidence_recall(retrieved, df["evidence"].tolist())
        enc = encode_pairs(tok, df["claim"].tolist(), texts, "vi", cfg.max_length)
        data = {**enc, "labels": torch.tensor(df["label"].values, dtype=torch.long)}
        torch.save(data, f"{cfg.data_dir}/tensors/vi_test_bm25_k{k}.pt")
        summary[f"k{k}"] = {"evidence_recall": rec}
        log(f"[bm25] k={k}: gold-evidence recall = {rec:.4f}")
    save_json(summary, f"{cfg.data_dir}/bm25_summary.json")


def load_tensors(cfg: Config, names: Iterable[str]) -> Dict[str, Dict]:
    import torch

    out = {}
    for n in names:
        p = f"{cfg.data_dir}/tensors/{n}.pt"
        if os.path.exists(p):
            out[n] = torch.load(p, weights_only=False)
    return out


def extra_parallel_sets(cfg: Config) -> List[str]:
    """Additional parallel test sets, e.g. par_ood_vifc (ViFactCheck test VI + its EN translation)."""
    d = f"{cfg.data_dir}/tensors"
    if not os.path.isdir(d):
        return []
    return sorted(f[:-3] for f in os.listdir(d) if f.startswith("par_ood_"))


def build_vifc_parallel_test(cfg: Config, log: Logger | None = None, mt: str = "vinai",
                             **kw) -> None:
    """EXP-9 entry point. mt='vinai' (default): offline VinAI Translate vi->en on the GPU,
    no rate limits. mt='google': Google Translate through deep-translator (often blocked on Colab)."""
    if mt == "vinai":
        return _build_vifc_vinai(cfg, log, **kw)
    return _build_vifc_google(cfg, log, **kw)


_SENT = None


def _split_sentences(text: str) -> List[str]:
    import re

    global _SENT
    _SENT = _SENT or re.compile(r"(?<=[.!?])\s+|\n+")
    parts = [p.strip() for p in _SENT.split(str(text)) if p and p.strip()]
    return parts or [str(text)]


def _build_vifc_vinai(cfg: Config, log: Logger | None = None, batch_size: int = 32,
                      num_beams: int = 5) -> None:
    """VinAI Translate (Nguyen et al., INTERSPEECH 2022), model vinai/vinai-translate-vi2en-v2.
    Evidence is translated sentence by sentence and re-joined, so long passages are not cut."""
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    log = log or Logger(f"{cfg.data_dir}/prepare.log")
    df = load_splits(cfg)["vi_test"].reset_index(drop=True)
    name = "vinai/vinai-translate-vi2en-v2"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mtok = AutoTokenizer.from_pretrained(name, src_lang="vi_VN")
    model = AutoModelForSeq2SeqLM.from_pretrained(name).to(device).eval()
    if device.type == "cuda":
        model = model.half()
    en_id = mtok.convert_tokens_to_ids("en_XX")

    # all unique segments (claims + evidence sentences), translated once
    claims = df["claim"].astype(str).tolist()
    ev_sents = [_split_sentences(e) for e in df["evidence"].astype(str)]
    segs = sorted(set(claims) | {x for ss in ev_sents for x in ss}, key=len)
    log(f"[vi2en/vinai] {len(segs)} unique segments on {device}")
    out: Dict[str, str] = {}
    with torch.no_grad():
        for i in range(0, len(segs), batch_size):
            batch = segs[i:i + batch_size]
            enc = mtok(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
            ids = model.generate(**enc, decoder_start_token_id=en_id, num_beams=num_beams,
                                 early_stopping=True, max_new_tokens=512)
            for src, tgt in zip(batch, mtok.batch_decode(ids, skip_special_tokens=True)):
                out[src] = tgt
            if (i // batch_size) % 20 == 0:
                log(f"[vi2en/vinai] {min(i + batch_size, len(segs))}/{len(segs)}")
    tr = pd.DataFrame({"idx": range(len(df)),
                       "claim_en": [out[c] for c in claims],
                       "evidence_en": [" ".join(out[x] for x in ss) for ss in ev_sents]})
    bad = int(((tr["claim_en"].str.strip() == "") | (tr["evidence_en"].str.strip() == "")).sum())
    if bad:
        raise RuntimeError(f"{bad} empty translations")
    tr.to_csv(f"{cfg.data_dir}/vifc_test_vi2en.csv", index=False)
    del model
    _write_vifc_tensors(cfg, df, tr, log, translator=f"{name} (beam {num_beams}, sentence-level for evidence)")


def _build_vifc_google(cfg: Config, log: Logger | None = None,
                       delay: float = 1.2, max_consecutive_fail: int = 6) -> None:
    """EXP-9: ViFactCheck test (native Vietnamese, gold evidence) machine-translated VI->EN.
    Gives a parallel test set in the news domain, which no system sees during training.

    Resumable and rate-limit aware: requests are spaced by `delay` seconds, a rate-limit error
    triggers exponential back-off (30 s, 60 s, 120 s, ...), and after `max_consecutive_fail`
    failed items the function saves what it has and stops; run it again later to continue.
    Only successful translations are cached (keyed by row index)."""
    import datetime as dt
    import time

    import torch
    from deep_translator import GoogleTranslator

    log = log or Logger(f"{cfg.data_dir}/prepare.log")
    df = load_splits(cfg)["vi_test"].reset_index(drop=True)
    cache = f"{cfg.data_dir}/vifc_test_vi2en.csv"
    tr = GoogleTranslator(source="vi", target="en")
    done: Dict[int, Dict] = {}
    if os.path.exists(cache):
        old = pd.read_csv(cache, keep_default_na=False)
        for r in old.to_dict("records"):
            if str(r["claim_en"]).strip() and str(r["evidence_en"]).strip():
                done[int(r["idx"])] = r
    log(f"[vi2en] cached: {len(done)}/{len(df)}")

    def translate(text: str) -> str:
        text = str(text)[:4500]
        if not text.strip():
            return text
        for attempt in range(5):
            try:
                time.sleep(delay)
                out = tr.translate(text)
                if out and out.strip():
                    return out
            except Exception as e:  # rate limit / network
                wait = 30 * 2 ** attempt if "too many requests" in str(e).lower() else 5 * (attempt + 1)
                log(f"[vi2en] attempt {attempt}: {str(e)[:60]}... waiting {wait}s")
                time.sleep(wait)
        return ""

    def save():
        pd.DataFrame([done[k] for k in sorted(done)]).to_csv(cache, index=False)

    fails = 0
    for i in range(len(df)):
        if i in done:
            continue
        c = translate(df.loc[i, "claim"])
        e = translate(df.loc[i, "evidence"]) if c else ""
        if c and e:
            done[i] = {"idx": i, "claim_en": c, "evidence_en": e}
            fails = 0
        else:
            fails += 1
            if fails >= max_consecutive_fail:
                save()
                raise RuntimeError(f"Stopped after {fails} consecutive failures (Google rate limit). "
                                   f"{len(done)}/{len(df)} translated and cached. Wait 30-60 min "
                                   f"(or restart the Colab runtime) and run the same command again.")
        if len(done) % 50 == 0:
            save()
            log(f"[vi2en] {len(done)}/{len(df)}")
    save()
    missing = [i for i in range(len(df)) if i not in done]
    if missing:
        raise RuntimeError(f"{len(missing)} rows still untranslated; run the same command again.")
    out = pd.DataFrame([done[i] for i in range(len(df))])
    _write_vifc_tensors(cfg, df, out, log, translator="Google Translate via deep-translator (vi->en)")


def _write_vifc_tensors(cfg: Config, df: pd.DataFrame, out: pd.DataFrame, log: Logger,
                        translator: str) -> None:
    import datetime as dt

    import torch

    tok = get_tokenizer(cfg)
    en = encode_pairs(tok, out["claim_en"].tolist(), out["evidence_en"].tolist(), "en", cfg.max_length)
    vi = encode_pairs(tok, df["claim"].tolist(), df["evidence"].tolist(), "vi", cfg.max_length)
    data = {"en_input_ids": en["input_ids"], "en_attention_mask": en["attention_mask"],
            "vi_input_ids": vi["input_ids"], "vi_attention_mask": vi["attention_mask"],
            "labels": torch.tensor(df["label"].values, dtype=torch.long)}
    torch.save(data, f"{cfg.data_dir}/tensors/par_ood_vifc.pt")
    save_json({"translator": translator,
               "date_utc": dt.datetime.utcnow().isoformat(timespec="seconds"),
               "n": len(out), "source": "ViFactCheck test (claim + gold evidence), vi->en"},
              f"{cfg.data_dir}/vifc_translation_meta.json")
    log(f"[vi2en] par_ood_vifc: {len(out)} pairs written ({translator})")


def extra_eval_sets(cfg: Config) -> List[str]:
    d = f"{cfg.data_dir}/tensors"
    if not os.path.isdir(d):
        return []
    return sorted(f[:-3] for f in os.listdir(d) if f.startswith("vi_test_bm25_"))
