"""Part 1 - Machine translation of FEVER pairs (only when no legacy corpus is reused).

Records the translator, date and failures in translation_meta.json so the
Limitations section can state exactly how the parallel corpus was produced.
"""
from __future__ import annotations

import datetime as dt
import os
import time

import pandas as pd

from .utils import Logger, save_json

MAX_CHARS = 4500  # deep-translator/Google limit is 5000 characters per request


def translate_pairs(src: pd.DataFrame, out_path: str, log: Logger,
                    save_every: int = 200) -> pd.DataFrame:
    from deep_translator import GoogleTranslator

    tr = GoogleTranslator(source="en", target="vi")
    done = pd.read_json(out_path) if os.path.exists(out_path) else pd.DataFrame()
    have = set(done["claim_en"]) if len(done) else set()
    rows = done.to_dict("records") if len(done) else []
    failures, truncated = 0, 0

    for _, r in src.iterrows():
        if r["claim"] in have:
            continue
        claim, ev = r["claim"], r["evidence"]
        if len(ev) > MAX_CHARS:
            truncated += 1
            ev = ev[:MAX_CHARS]
        for attempt in range(3):
            try:
                rows.append({"claim_en": r["claim"], "evidence_en": r["evidence"],
                             "claim_vi": tr.translate(claim), "evidence_vi": tr.translate(ev),
                             "label": int(r["label"])})
                break
            except Exception as e:  # network/rate limit
                time.sleep(2 * (attempt + 1))
                if attempt == 2:
                    failures += 1
                    log(f"[mt] failed: {e}")
        if len(rows) % save_every == 0:
            pd.DataFrame(rows).to_json(out_path, orient="records", force_ascii=False)

    out = pd.DataFrame(rows)
    out.to_json(out_path, orient="records", force_ascii=False)
    save_json({
        "translator": "Google Translate via deep-translator (GoogleTranslator, en->vi)",
        "date_utc": dt.datetime.utcnow().isoformat(timespec="seconds"),
        "requested": len(src), "translated": len(out), "failures": failures,
        "evidence_truncated_to_chars": MAX_CHARS, "n_truncated": truncated,
    }, os.path.join(os.path.dirname(out_path), "translation_meta.json"))
    log(f"[mt] {len(out)} pairs translated, {failures} failures, {truncated} truncated")
    return out
