"""EXP-10: explainability analyses (evidence reliance, IG faithfulness, cross-lingual attributions).

Needs best.pt of the runs to analyse. The EXP-9 re-runs can keep them:
    python scripts/ood_consistency.py --root $ROOT --train --save-best \
        --variants full mt_xlmr_vi translate_train
Then:
    python scripts/run_explain.py --root $ROOT --runs-subdir runs_ood \
        --variants full mt_xlmr_vi --seeds 42 123 456 --n-ig 200
    python scripts/run_explain.py --root $ROOT --runs-subdir runs_ood --report
Smoke test on CPU (tiny random model, ~1 min):
    python scripts/run_explain.py --smoke
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import glob
import json
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xalignfact.config import make_config  # noqa: E402
from xalignfact.evaluate import clcs_metrics, classification_metrics, predict, predict_parallel  # noqa: E402
from xalignfact.explain import SpecialIds, ablate_segment, explain_parallel, explain_single  # noqa: E402
from xalignfact.model import EN, VI  # noqa: E402


def probes(model, data, sp, device):
    """Accuracy / consistency with full input, claim only and evidence only."""
    out = {}
    for name, lang in (("vi_test", VI), ("en_test", EN)):
        if name not in data:
            continue
        for cond in ("full", "claim", "evidence"):
            d = data[name] if cond == "full" else ablate_segment(data[name], sp, cond)
            p = predict(model, d, lang, device)
            out[f"{name}_{cond}_f1"] = classification_metrics(p["labels"], p["preds"])["macro_f1"]
    for name in [n for n in ("par_test", "par_ood_vifc") if n in data]:
        for cond in ("full", "claim", "evidence"):
            d = data[name]
            if cond != "full":
                d = ablate_segment(ablate_segment(d, sp, cond, "en_"), sp, cond, "vi_")
            m = clcs_metrics(predict_parallel(model, d, device))
            out[f"{name}_{cond}_clcs"] = m["clcs"]
            out[f"{name}_{cond}_joint"] = m["joint_accuracy"]
    return out


def analyse(model, data, sp, tok, device, n_ig, steps):
    res = {"probes": probes(model, data, sp, device)}
    for name, lang in (("vi_test", VI), ("en_test", EN)):
        if name in data:
            res[f"ig_{name}"] = explain_single(model, data[name], lang, sp, n_ig, steps)
    for name in ("par_test", "par_ood_vifc"):
        if name in data:
            res[f"xl_{name}"] = explain_parallel(model, data[name], sp, tok, n_ig, steps)
    return res


def smoke():
    from types import SimpleNamespace

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests"))
    from smoke_test import L, VOCAB, tiny_encoder

    from xalignfact.model import XAlignFact

    cfg = make_config("full", seed=0)
    model = XAlignFact(cfg, encoder_module=tiny_encoder()).eval()
    sp = SpecialIds(cls=0, sep=2, pad=1, mask=3)
    g = torch.Generator().manual_seed(0)

    def seq(n):
        ids = torch.randint(5, VOCAB, (n, L), generator=g)
        mask = torch.ones(n, L, dtype=torch.long)
        ids[:, 0], ids[:, 6], ids[:, 7], ids[:, L - 4] = 0, 2, 2, 2
        ids[:, L - 3:] = 1
        mask[:, L - 3:] = 0
        return ids, mask

    a, b = seq(12)
    c, d = seq(12)
    y = torch.randint(0, 3, (12,), generator=g)
    data = {"vi_test": {"input_ids": a, "attention_mask": b, "labels": y},
            "en_test": {"input_ids": c, "attention_mask": d, "labels": y},
            "par_test": {"en_input_ids": c, "en_attention_mask": d, "vi_input_ids": a,
                         "vi_attention_mask": b, "labels": y}}
    cl = ablate_segment(data["vi_test"], sp, "claim")
    assert int(cl["attention_mask"][0].sum()) == 7, "claim-only should be <s> [L] c c c c </s>"
    ev = ablate_segment(data["vi_test"], sp, "evidence")
    assert int(ev["attention_mask"][0].sum()) == L - 3 - 4, "evidence-only length"
    tok = SimpleNamespace(convert_ids_to_tokens=lambda ids: [str(i % 10) if i % 7 == 0 else "w" for i in ids])
    res = analyse(model, data, sp, tok, torch.device("cpu"), n_ig=4, steps=4)
    print(json.dumps(res, indent=1)[:1500])
    # completeness check of IG: sum of attributions ~ p(x) - p(baseline) for many steps
    print("smoke test passed")


def report(root, sub):
    rows = []
    for p in glob.glob(f"{root}/{sub}/*/seed*/explain.json"):
        r = json.load(open(p))
        flat = {"variant": r["variant"], "seed": r["seed"], **r["probes"]}
        for k, v in r.items():
            if k.startswith(("ig_", "xl_")) and isinstance(v, dict):   # skip ig_steps etc.
                for kk, vv in v.items():
                    if isinstance(vv, dict):
                        for k3, v3 in vv.items():
                            flat[f"{k}.{kk}.{k3}"] = v3
                    else:
                        flat[f"{k}.{kk}"] = vv
        rows.append(flat)
    if not rows:
        print("no explain.json found")
        return
    df = pd.DataFrame(rows)
    os.makedirs(f"{root}/results", exist_ok=True)
    df.to_csv(f"{root}/results/explain_per_run.csv", index=False)
    summ = df.drop(columns="seed").groupby("variant").agg(["mean", "std"]).T.round(4)
    summ.to_csv(f"{root}/results/explain_summary.csv")
    pd.set_option("display.max_rows", 200)
    print(summ.to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root")
    ap.add_argument("--runs-subdir", default="runs_ood")
    ap.add_argument("--variants", nargs="*", default=["full", "mt_xlmr_vi"])
    ap.add_argument("--seeds", nargs="*", type=int, default=[42, 123, 456])
    ap.add_argument("--n-ig", type=int, default=200)
    ap.add_argument("--ig-steps", type=int, default=32)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    if a.smoke:
        smoke()
        sys.exit()
    if a.report:
        report(a.root, a.runs_subdir)
        sys.exit()

    from xalignfact.analysis import load_model
    from xalignfact.data import get_tokenizer, load_tensors
    from xalignfact.utils import load_json, save_json

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for v in a.variants:
        for s in a.seeds:
            cfg = make_config(v, seed=s, root=a.root, runs_subdir=a.runs_subdir)
            ck = f"{cfg.run_dir}/best.pt"
            if not os.path.exists(ck):
                print(f"[skip] {ck} not found (re-run with --save-best)")
                continue
            if os.path.exists(f"{cfg.run_dir}/explain.json"):
                print(f"[skip] {v}/seed{s}: explain.json exists")
                continue
            tok = get_tokenizer(cfg)
            sp = SpecialIds.from_tokenizer(tok)
            vocab = load_json(f"{cfg.data_dir}/manifest.json")["vocab_size"]
            data = load_tensors(cfg, ["vi_test", "en_test", "par_test", "par_ood_vifc"])
            model = load_model(cfg, ck, vocab, device)
            res = analyse(model, data, sp, tok, device, a.n_ig, a.ig_steps)
            res.update({"variant": v, "seed": s, "n_ig": a.n_ig, "ig_steps": a.ig_steps})
            save_json(res, f"{cfg.run_dir}/explain.json")
            print(f"[done] {v}/seed{s}: {res['probes']}")
            del model
            torch.cuda.empty_cache()
    report(a.root, a.runs_subdir)
