"""Build the Zenodo data/results archive for the v2 release (run on Colab after all experiments).

    python scripts/package_release.py --root $ROOT --annotated /path/flagged_pairs_annotated.csv

Writes /content/xalignfact_v2_data_results.zip (or --out). Licence: CC BY-SA 4.0, because the
parallel corpus, splits and predictions are derived from FEVER. The code itself is archived
separately through the GitHub-Zenodo integration (MIT).

Included
  data/parallel_en_vi_v2.json                     EN->VI translations of FEVER pairs (Google)
  data/splits/{en_*,par_*}.jsonl                  FEVER-derived splits (text + claim hashes)
  data/manifest.json, data/overlap_report.json    split hashes and leakage checks (all splits)
  data/translation_quality.csv                    negation/number flags and LaBSE per test pair
  data/flagged_pairs_annotated.csv                manual check of the 58 flagged pairs
  data/bm25_summary.json, data/vifc_translation_meta.json   settings only, no ViFactCheck text
  results/** (csv, json, figures)                 aggregated tables and figures
  runs*/<variant>/seed*/{results,train_log,config,env}.json, train.log
  runs*/<variant>/seed*/preds_*.npz               predictions; for ViFactCheck-derived sets the
                                                  gold labels are removed (rebuild them with
                                                  scripts/prepare_data.py from the official release)
Excluded
  ViFactCheck text in any form (data/splits/vi_*.jsonl, data/vifc_test_vi2en.csv, tensors),
  tokenised tensors, model checkpoints (best.pt, ckpt_*.pt).
"""
import argparse
import glob
import io
import os
import zipfile

import numpy as np

LICENCE_TXT = """X-AlignFact v2 - data, predictions and logs
Licence: Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)
https://creativecommons.org/licenses/by-sa/4.0/

Derived from FEVER (Thorne et al., NAACL 2018; CC BY-SA) in its NLI-style release
(Nie, Chen and Bansal, AAAI 2019; pietrolesci/nli_fever). The English-to-Vietnamese
translations were produced automatically (Google Translate via deep-translator) without
post-editing. ViFactCheck (Hoa et al., AAAI 2025) is NOT redistributed: predictions on
ViFactCheck-derived sets contain only predicted labels, indexed as in the official test
split; rebuild the data with scripts/prepare_data.py from the official release.
Code: MIT licence, archived separately (GitHub release v2.0).
"""

VI_DERIVED = ("vi_test", "vi_test_en_head", "vi_test_bm25", "par_ood_vifc")


def add(z, path, arc):
    if os.path.exists(path):
        z.write(path, arc)
        return 1
    return 0


def add_preds(z, path, arc):
    name = os.path.basename(path)[len("preds_"):-len(".npz")]
    if name.startswith(VI_DERIVED):
        d = dict(np.load(path))
        d.pop("labels", None)                      # no ViFactCheck gold labels
        buf = io.BytesIO()
        np.savez_compressed(buf, **d)
        z.writestr(arc, buf.getvalue())
    else:
        z.write(path, arc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--annotated", help="flagged_pairs_annotated.csv (manual check of 58 pairs)")
    ap.add_argument("--out", default="/content/xalignfact_v2_data_results.zip")
    a = ap.parse_args()
    r = a.root
    n = 0
    with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("LICENSE.txt", LICENCE_TXT)
        for f in ("parallel_en_vi_v2.json", "manifest.json", "overlap_report.json",
                  "translation_quality.csv", "bm25_summary.json", "vifc_translation_meta.json"):
            n += add(z, f"{r}/data/{f}", f"data/{f}")
        for p in sorted(glob.glob(f"{r}/data/splits/*.jsonl")):
            b = os.path.basename(p)
            if b.startswith(("en_", "par_")):
                n += add(z, p, f"data/splits/{b}")
        if a.annotated:
            n += add(z, a.annotated, "data/flagged_pairs_annotated.csv")
        for p in sorted(glob.glob(f"{r}/results/**/*", recursive=True)):
            if os.path.isfile(p) and p.endswith((".csv", ".json", ".pdf", ".png")):
                n += add(z, p, os.path.relpath(p, r))
        for p in sorted(glob.glob(f"{r}/runs*/*/seed*/*")):
            b = os.path.basename(p)
            arc = os.path.relpath(p, r)
            if b in ("results.json", "train_log.json", "config.json", "env.json", "train.log",
                     "explain.json"):
                n += add(z, p, arc)
            elif b.startswith("preds_") and b.endswith(".npz"):
                add_preds(z, p, arc)
                n += 1
    mb = os.path.getsize(a.out) / 2**20
    print(f"{n} files -> {a.out} ({mb:.1f} MB)")
    with zipfile.ZipFile(a.out) as z:
        bad = [x for x in z.namelist() if x.startswith("data/splits/vi_") or "vifc_test_vi2en" in x
               or x.endswith(".pt")]
    assert not bad, f"ViFactCheck text or checkpoints in archive: {bad[:5]}"
    print("check passed: no ViFactCheck text, no tensors or checkpoints")


if __name__ == "__main__":
    main()
