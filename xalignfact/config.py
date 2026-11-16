"""Part 0 - Configuration and experiment variants.

Every run is fully described by a `Config`. The config is saved next to each
run's results, so any number in the paper can be traced to (variant, seed,
config, commit).

Changes vs. the submitted version (OJCS-2026-05-0480):
  * L_proto_cross removed (it had no gradient: prototypes are EMA buffers).
  * Curriculum re-implemented: the per-step batch composition follows the
    phase weights, and the classification loss is pooled over all labelled
    examples, so the sampling proportions set each language's gradient share.
  * Model selection uses dev data only (ViFactCheck dev + a parallel dev set).
  * All variants (main model, ablations, baselines) share one training pipeline.
"""
from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

LABELS = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
LABEL2ID = {name: i for i, name in enumerate(LABELS)}

DEFAULT_PHASES = [
    # end_epoch is inclusive; weights are (w_en, w_vi, w_par)
    {"name": "phase1", "end_epoch": 4, "weights": [0.5, 0.3, 0.2]},
    {"name": "phase2", "end_epoch": 9, "weights": [0.3, 0.4, 0.3]},
    {"name": "phase3", "end_epoch": 14, "weights": [0.2, 0.5, 0.3]},
]

DEFAULT_SEEDS = [42, 123, 456, 789, 2024]


@dataclass
class Config:
    # ---- identity -------------------------------------------------------
    variant: str = "full"
    seed: int = 42

    # ---- paths ----------------------------------------------------------
    root: str = "/content/drive/MyDrive/xalignfact_v2"

    # ---- data -----------------------------------------------------------
    fever_hf: str = "pietrolesci/nli_fever"
    # Which nli_fever column holds the claim. "auto" picks the column with the
    # shorter median length (FEVER claims are short) and logs the decision.
    fever_claim_field: str = "auto"
    fever_label_field: str = "fever_gold_label"
    fever_train_per_class: int = 30445
    fever_dev_max: int = 6000          # stratified subsample of nli_fever dev (monitoring only)
    vifact_hf: str = "tranthaihoa/vifactcheck"
    # Existing translated pairs (parallel_en_vi.json from v1). If given, these
    # pairs are reused and their claims are EXCLUDED from the English training pool.
    parallel_legacy_path: str = ""
    parallel_n_new: int = 5000         # only used when no legacy file is given
    parallel_dev_n: int = 500          # model selection (CLCS-dev)
    parallel_test_n: int = 1000        # reporting (CLCS-test)
    data_seed: int = 42                # fixed for all runs: splits never depend on the run seed
    max_length: int = 256
    bm25_topk: List[int] = field(default_factory=lambda: [1, 3, 5])

    # ---- model ----------------------------------------------------------
    encoder: str = "xlm-roberta-base"
    num_labels: int = 3
    head_dropout: float = 0.1
    shared_head: bool = False

    # ---- streams & losses ----------------------------------------------
    use_en: bool = True
    use_vi: bool = True
    use_parallel: bool = True
    parallel_ce: bool = False          # translate-train: CE on both sides of parallel pairs
    w_cons: float = 0.3
    w_repr: float = 0.3
    w_proto: float = 0.3
    w_infonce: float = 0.0
    temperature_proto: float = 0.10
    temperature_infonce: float = 0.10
    infonce_mask_same_label: bool = False
    ema_decay: float = 0.999

    # ---- curriculum -----------------------------------------------------
    phases: List[Dict] = field(default_factory=lambda: copy.deepcopy(DEFAULT_PHASES))
    step_budget: int = 64              # examples per step; a parallel pair counts as one example
    steps_per_epoch: int = 0           # 0 -> ceil(n_vi_train / 32), as in v1 (159 for ViFactCheck)

    # ---- optimisation ---------------------------------------------------
    lr: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.10
    num_epochs: int = 15
    patience: int = 5
    grad_clip: float = 1.0
    bf16: bool = True
    eval_batch_size: int = 128

    # ---- model selection -------------------------------------------------
    # "combined": 0.7 * VI-dev macro-F1 + 0.3 * CLCS-dev ; "vi_f1" ; "en_f1"
    selection: str = "combined"
    selection_vi_weight: float = 0.7

    # ---- saving -----------------------------------------------------------
    save_best_checkpoint: bool = False   # ~1.1 GB per run; enable for runs you need later
    save_phase_checkpoints: bool = False # needed only for the gradient analysis (EXP-7)
    runs_subdir: str = "runs"            # "runs_ood" for the out-of-domain re-runs (EXP-9)

    # ---------------------------------------------------------------------
    def to_dict(self) -> Dict:
        return asdict(self)

    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, path: str) -> "Config":
        with open(path, encoding="utf-8") as f:
            return cls(**json.load(f))

    def phase_for_epoch(self, epoch: int) -> Dict:
        for ph in self.phases:
            if epoch <= ph["end_epoch"]:
                return ph
        return self.phases[-1]

    def phase_weights(self, epoch: int) -> List[float]:
        """Phase weights (w_en, w_vi, w_par), normalised to sum to 1."""
        w = list(self.phase_for_epoch(epoch)["weights"])
        s = sum(w)
        if s <= 0:
            raise ValueError("Phase weights must not all be zero.")
        return [wi / s for wi in w]

    def stream_mask(self) -> List[bool]:
        return [self.use_en, self.use_vi, self.use_parallel]

    @property
    def run_dir(self) -> str:
        return f"{self.root}/{self.runs_subdir}/{self.variant}/seed{self.seed}"

    @property
    def data_dir(self) -> str:
        return f"{self.root}/data"


# ---------------------------------------------------------------------------
# Variants. Names are the ones used in code, tables and the response letter.
# Each entry maps to the ablation/baseline IDs of the revision plan.
# ---------------------------------------------------------------------------
def _flat_phases(weights):
    return [{"name": "flat", "end_epoch": 10**6, "weights": list(weights)}]


VARIANTS: Dict[str, Dict] = {
    # --- proposed model -------------------------------------------------
    "full": {},
    # --- ablations (EXP-3) ----------------------------------------------
    "no_cons": {"w_cons": 0.0},
    "no_repr": {"w_repr": 0.0},
    "no_proto": {"w_proto": 0.0},
    "no_parallel": {"use_parallel": False, "w_cons": 0.0, "w_repr": 0.0},
    "translate_train": {"parallel_ce": True, "w_cons": 0.0, "w_repr": 0.0},
    "plus_infonce": {"w_infonce": 0.3},
    "plus_infonce_labelaware": {"w_infonce": 0.3, "infonce_mask_same_label": True},
    # Curriculum controls: same compute, same losses, different batch composition.
    "flat_curriculum": {"phases": _flat_phases([1 / 3, 1 / 3, 1 / 3])},
    # v1 effectively used a constant 32 EN : 32 VI : 16 pairs composition.
    "fixed_ratio": {"phases": _flat_phases([0.4, 0.4, 0.2])},
    "shared_head": {"shared_head": True},
    # Follow-up ablations suggested by the first round of results (report as such):
    "cons_only": {"w_repr": 0.0, "w_proto": 0.0},                       # L_task + L_cons
    "translate_train_cons": {"parallel_ce": True, "w_repr": 0.0, "w_proto": 0.0},  # CE on pairs + L_cons
    # --- baselines (EXP-6), trained with the same pipeline ----------------
    "mono_en": {"use_vi": False, "use_parallel": False, "w_cons": 0.0, "w_repr": 0.0,
                "w_proto": 0.0, "selection": "en_f1",
                "phases": _flat_phases([1.0, 0.0, 0.0])},
    "mono_vi": {"use_en": False, "use_parallel": False, "w_cons": 0.0, "w_repr": 0.0,
                "w_proto": 0.0, "selection": "vi_f1",
                "phases": _flat_phases([0.0, 1.0, 0.0])},
    # Same selection rule as every two-head model (0.7 VI-dev F1 + 0.3 CLCS-dev).
    "mt_xlmr": {"use_parallel": False, "w_cons": 0.0, "w_repr": 0.0, "w_proto": 0.0,
                "phases": _flat_phases([0.5, 0.5, 0.0])},
    # MT-XLM-R selected on VI-dev F1 only. The combined rule (0.7 F1 + 0.3 CLCS) rewards a
    # property this baseline is not trained for and stopped some runs early; selecting the
    # baseline on its own best VI checkpoint makes the F1 comparison conservative for X-AlignFact.
    "mt_xlmr_vi": {"use_parallel": False, "w_cons": 0.0, "w_repr": 0.0, "w_proto": 0.0,
                   "phases": _flat_phases([0.5, 0.5, 0.0]), "selection": "vi_f1"},
}

# Note: v1's "equal loss weights" variant (A7) is not reproduced. With
# L_proto_cross removed, the default alignment weights are already equal (0.3 each).


def make_config(variant: str = "full", seed: int = 42, **overrides) -> Config:
    if variant not in VARIANTS:
        raise KeyError(f"Unknown variant '{variant}'. Known: {sorted(VARIANTS)}")
    params = copy.deepcopy(VARIANTS[variant])
    params.update(overrides)
    return Config(variant=variant, seed=seed, **params)
