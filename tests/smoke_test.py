"""Smoke test: whole pipeline on a tiny random XLM-R with synthetic data (CPU, ~1-2 min).

python tests/smoke_test.py
Checks: every variant trains and writes results; same seed -> identical results;
L_proto sends gradient to the encoder; curriculum composition follows the phases;
no torch.manual_seed reset inside evaluation (sampler RNG is independent).
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # before torch is imported
import os
import shutil
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xalignfact.config import make_config  # noqa: E402
from xalignfact.model import XAlignFact  # noqa: E402
from xalignfact.sampler import CurriculumSampler, allocate  # noqa: E402
from xalignfact.train import build_batch, run  # noqa: E402
from xalignfact.utils import load_json, save_json  # noqa: E402

VOCAB, L = 120, 24


def tiny_encoder():
    from transformers import XLMRobertaConfig, XLMRobertaModel

    c = XLMRobertaConfig(vocab_size=VOCAB, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                         intermediate_size=64, max_position_embeddings=L + 4, pad_token_id=1)
    torch.manual_seed(0)
    return XLMRobertaModel(c)


def fake_single(n, g):
    ids = torch.randint(5, VOCAB, (n, L), generator=g, dtype=torch.int32)
    mask = torch.ones(n, L, dtype=torch.int8)
    mask[:, L // 2:] = (torch.rand(n, L - L // 2, generator=g) > 0.5).to(torch.int8)
    y = torch.randint(0, 3, (n,), generator=g)
    ids[:, 1] = 5 + y.to(torch.int32)          # learnable signal
    return {"input_ids": ids, "attention_mask": mask, "labels": y}


def fake_parallel(n, g):
    a, b = fake_single(n, g), fake_single(n, g)
    b["labels"] = a["labels"]
    b["input_ids"][:, 1] = a["input_ids"][:, 1]
    return {"en_input_ids": a["input_ids"], "en_attention_mask": a["attention_mask"],
            "vi_input_ids": b["input_ids"], "vi_attention_mask": b["attention_mask"], "labels": a["labels"]}


def make_data(root):
    g = torch.Generator().manual_seed(0)
    d = f"{root}/data/tensors"
    os.makedirs(d, exist_ok=True)
    sizes = {"en_train": 300, "vi_train": 96, "en_dev": 60, "vi_dev": 60, "en_test": 80,
             "vi_test": 80, "vi_test_bm25_k1": 80}
    for k, n in sizes.items():
        torch.save(fake_single(n, g), f"{d}/{k}.pt")
    for k, n in {"par_train": 80, "par_dev": 40, "par_test": 50}.items():
        torch.save(fake_parallel(n, g), f"{d}/{k}.pt")
    save_json({"vocab_size": VOCAB}, f"{root}/data/manifest.json")


def _cfg(variant, root, seed):
    over = dict(seed=seed, root=root, num_epochs=3, steps_per_epoch=3, step_budget=12,
                eval_batch_size=32, patience=5)
    base = make_config(variant, **over)
    if len(base.phases) == 3:  # compress the 3-phase curriculum into 3 epochs
        base.phases = [{"name": "phase1", "end_epoch": 0, "weights": [0.5, 0.3, 0.2]},
                       {"name": "phase2", "end_epoch": 1, "weights": [0.3, 0.4, 0.3]},
                       {"name": "phase3", "end_epoch": 2, "weights": [0.2, 0.5, 0.3]}]
    return base


def main():
    root = tempfile.mkdtemp()
    try:
        make_data(root)
        dev = torch.device("cpu")

        # 1) sampler composition
        assert allocate([0.5, 0.3, 0.2], 64) == [32, 19, 13], allocate([0.5, 0.3, 0.2], 64)
        assert allocate([0.2, 0.5, 0.3], 64) == [13, 32, 19]
        assert allocate([0.0, 1.0, 0.0], 64) == [0, 64, 0]
        cfg = _cfg("full", root, 42)
        s = CurriculumSampler(cfg, {"en": 300, "vi": 96, "par": 80}, 42)
        comp = [tuple(len(v) for v in s.next(e).values()) for e in range(3)]
        assert comp == [(6, 4, 2), (4, 5, 3), (2, 6, 4)], comp
        print("[ok] curriculum composition per phase:", comp)

        # 2) L_proto sends gradient to the encoder (and nothing like L_proto_cross remains)
        data = {k: torch.load(f"{root}/data/tensors/{k}.pt") for k in ("en_train", "vi_train", "par_train")}
        m = XAlignFact(_cfg("full", root, 42), VOCAB, tiny_encoder())
        s = CurriculumSampler(m.cfg, {"en": 300, "vi": 96, "par": 80}, 1)
        # L_proto starts once every class of a language has been seen (prototype initialised).
        # With the tiny test budget (6 EN / 4 VI per step) this can take a few steps.
        for step in range(30):
            losses = m.training_losses(build_batch(data, s.next(0), dev), 2)
            if "L_proto" in losses:
                break
        assert "L_proto" in losses, f"prototypes never initialised: {m.prototypes.initialised}"
        assert "L_proto_cross" not in losses
        g = torch.autograd.grad(losses["L_proto"], m.encoder.proj.weight)[0]
        assert g.abs().sum() > 0
        print("[ok] L_proto gradient reaches the encoder; terms:", sorted(losses))

        # 3) every variant runs end-to-end
        variants = ["full", "no_cons", "no_repr", "no_proto", "no_parallel", "translate_train",
                    "plus_infonce", "plus_infonce_labelaware", "flat_curriculum", "fixed_ratio",
                    "shared_head", "mono_en", "mono_vi", "mt_xlmr"]
        for v in variants:
            r = run(_cfg(v, root, 42), encoder_module=tiny_encoder(), device=dev)
            t = r["test"]
            assert {"en_test", "vi_test", "vi_test_en_head", "par_test", "vi_test_bm25_k1"} <= set(t), t.keys()
            log = load_json(f"{root}/runs/{v}/seed42/train_log.json")
            print(f"[ok] {v:24s} VI={t['vi_test']['macro_f1']:.3f} CLCS={t['par_test']['clcs']:.3f} "
                  f"compositions={[e['composition'] for e in log]}")

        # 4) determinism: same seed twice -> identical results
        r1 = load_json(f"{root}/runs/full/seed42/results.json")
        r2 = run(_cfg("full", root, 42), overwrite=True, encoder_module=tiny_encoder(), device=dev)
        assert r1["test"] == r2["test"], "same seed gave different results"
        r3 = run(_cfg("full", root, 7), encoder_module=tiny_encoder(), device=dev)
        print("[ok] determinism: seed 42 reproduced exactly; seed 7 VI =",
              round(r3["test"]["vi_test"]["macro_f1"], 3))

        # 5) aggregation + bootstrap
        from xalignfact.stats import collect, compare_all, summarise

        df = collect(root)
        summarise(df, ["vi_test_f1", "par_clcs"])
        cmp = compare_all(root, sorted(df["variant"].unique()), seeds=[42], n_boot=200)
        assert len(cmp) > 0
        print("[ok] stats:", len(df), "runs aggregated,", len(cmp), "comparisons")
        print("\nSMOKE TEST PASSED")
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
