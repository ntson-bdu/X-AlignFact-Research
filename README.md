# X-AlignFact v2

Code for **"X-AlignFact: A Cross-Lingual Alignment Framework for Bilingual Fact Verification
in English and Vietnamese"** by Son Thanh Nguyen, Trong-Thua Huynh, Huu Lap Le and Tuan-Anh Le
(submitted to the IEEE Open Journal of the Computer Society, manuscript OJCS-2026-11-1196).

Bilingual (EN–VI) three-way claim–evidence verification with a shared XLM-R encoder,
language-specific heads and cross-lingual alignment on parallel pairs. This repository
(release `v2.0`) replaces the v1 notebook of the earlier submission OJCS-2026-05-0480.
Every number in the article is produced by the commands below.

- Code: this repository, MIT License (`LICENSE`).
- Archive of code, data, per-run logs and predictions: https://doi.org/10.5281/zenodo.23850412
  (data under CC BY-SA 4.0; see `LICENSES.md`).
- ViFactCheck is **not** redistributed; `scripts/prepare_data.py` rebuilds every split from the
  official release.

## Quick start (Colab, A100)

```bash
pip install -r requirements.txt
python tests/smoke_test.py                       # ~1-2 min on CPU, no downloads
ROOT=/content/drive/MyDrive/xalignfact_v2
python scripts/prepare_data.py --root $ROOT \
  --legacy-parallel /content/drive/MyDrive/xalignfact/data/parallel/parallel_en_vi.json
python scripts/run.py --root $ROOT --variants full --seeds 42 --save-best --save-phase-ckpts
python scripts/run.py --root $ROOT --group main          # remaining 4 seeds of the full model
python scripts/run.py --root $ROOT --group baselines     # mono_en, mono_vi, mt_xlmr
python scripts/baseline_phobert.py --root $ROOT          # PhoBERT-base (VI)
python scripts/run.py --root $ROOT --group ablations     # 8 variants x 5 seeds (~7 h)
python scripts/aggregate.py --root $ROOT                 # tables + paired bootstrap
python scripts/run_analysis.py --root $ROOT --seed 42
```

`notebooks/X_AlignFact_v2_Colab.ipynb` runs the same steps cell by cell.
Runs are resume-safe: a finished `runs/<variant>/seed<k>/results.json` is skipped.

## Layout (mirrors the v1 notebook parts)

| Part | File | Content |
| --- | --- | --- |
| 0 | `xalignfact/config.py`, `utils.py` | configuration, variants, seeding/determinism, environment capture |
| 1 | `xalignfact/data.py`, `translate.py` | leakage-free splits, overlap report, tokenisation, BM25 test sets |
| 2 | `xalignfact/model.py`, `losses.py` | encoder, heads, EMA prototypes, L_cons, L_repr, L_proto, (InfoNCE ablation) |
| 3 | `xalignfact/sampler.py`, `train.py` | per-step curriculum sampler; one training pipeline for every variant |
| 4 | `xalignfact/evaluate.py` | macro-F1, per-class, confusion counts, CLCS / CLCS-p / kappa / joint accuracy |
| 5 | `xalignfact/stats.py`, `scripts/aggregate.py` | mean ± SD over seeds, 95% t-interval, paired bootstrap, Holm |
| 6 | `xalignfact/analysis.py`, `retrieval.py` | t-SNE (test), error counts, gradient cosines per phase, translation sheet, BM25 |

## What changed from v1 and why

| v1 issue | v2 behaviour | Review item |
| --- | --- | --- |
| Main results and ablations came from two different training scripts; the ablation evaluator reset `torch.manual_seed(42)` every epoch | One `run(cfg)` for every variant; samplers use their own `torch.Generator`; evaluation consumes no training randomness | AE1, R1.1 |
| English "test" = 5% of the training pool | Labelled `nli_fever` development split (18,209 rows; shared-task test labels are not public); training rows sharing a claim with any evaluation split removed; claim-level overlap check | AE6b |
| Model selection on VI *train* F1 and on the CLCS reporting set; ablations selected on test | Selection on ViFactCheck dev + a separate parallel dev set; test evaluated once | AE2c, AE6b |
| Main results from one seed (reported as three) | 5 seeds (42, 123, 456, 789, 2024); mean ± SD, CI, paired tests | AE2a/b, R1.2 |
| `L_proto_cross` had no gradient (EMA buffers) | Removed | AE5a |
| Curriculum: constant 32 EN : 32 VI : 16 pairs per step | Per-step composition follows the phase weights (64-example budget); classification loss pooled over labelled examples | AE5a |
| InfoNCE and equal-weight variants were not configured as described | `plus_infonce` implemented; equal-weight variant dropped (alignment weights are already equal at 0.3) | AE5a |
| Hard negatives generated but unused; counterfactual XAI untrained | Both removed | AE6 |
| Retrieval modules described but absent | Removed; primary setting = evidence provided with the datasets (gold for ViFactCheck, automatically selected NLI-FEVER contexts for English); BM25 over ViFactCheck `Context` as a separate retrieved-evidence setting | AE3, R1.3 |
| Parallel pairs drawn from the English training pool | Parallel claims excluded from English train/dev/test; split into par_train / par_dev / par_test | AE4, AE6b |
| t-SNE on training data with prototypes from a separate fit; "symmetric 10/10 errors" was a loop cap | t-SNE on test with prototypes in the same fit; all errors counted | AE5b |

## Variants

| Name | Description |
| --- | --- |
| `full` | proposed model: L_task + 0.3 L_cons + 0.3 L_repr + 0.3 L_proto, 3-phase curriculum |
| `no_cons`, `no_repr`, `no_proto` | remove one alignment term |
| `no_parallel` | no parallel pairs (EN/VI per-step counts identical to `full`) |
| `translate_train` | parallel pairs used only as labelled data (CE on both sides), no L_cons/L_repr |
| `plus_infonce` | `full` + 0.3 symmetric cross-lingual InfoNCE (tau = 0.1) |
| `plus_infonce_labelaware` | as above, same-label negatives masked (optional) |
| `flat_curriculum` | constant 1/3 : 1/3 : 1/3 composition |
| `fixed_ratio` | constant 0.4 : 0.4 : 0.2 composition (v1's effective ratio) |
| `shared_head` | one head for both languages (optional) |
| `mono_en`, `mono_vi`, `mt_xlmr` | baselines: XLM-R on EN only / VI only / EN+VI without parallel data or alignment |
| `phobert_vi` | PhoBERT-base on ViFactCheck (`scripts/baseline_phobert.py`) |

All variants use the same compute: `steps_per_epoch = ceil(|VI train| / 32)` (159) steps per epoch,
a 64-example budget per step (a parallel pair counts as one example; disabled streams leave their
slots empty), 15 epochs, AdamW 5e-5, OneCycle (10% warm-up), early stopping (patience 5) on dev.
Selection: 0.7 × VI-dev macro-F1 + 0.3 × CLCS-dev for two-head models; VI-dev F1 for `mono_vi`,
EN-dev F1 for `mono_en`. `vi_test_en_head` reports the EN head on Vietnamese (zero-shot read-out for `mono_en`).

## Notes

- ViFactCheck label ids: 0 = SUPPORTS, 1 = REFUTES, 2 = NOT ENOUGH INFO.
- PhoBERT inputs are word-segmented with `underthesea`.
- Exact GPU determinism needs the same GPU/CUDA/library versions; each run writes `env.json`
  (including `pip freeze`) next to its results. The runs in the article used one NVIDIA A100-SXM4 40 GB.
- `scripts/check_numbers.py --root $ROOT` recomputes the numbers reported in the article from the
  per-run outputs; `scripts/make_figures.py` and `scripts/make_supp_tables.py` rebuild the figures
  and the Supplementary tables; `scripts/package_release.py` builds the Zenodo data archive.

## Outputs per run (`runs/<variant>/seed<k>/`)

`config.json`, `env.json`, `train_log.json` (per-epoch composition, losses, dev metrics),
`results.json` (test metrics incl. confusion counts), `preds_*.npz` (labels, predictions,
probabilities), optional `best.pt` and phase checkpoints. `data/manifest.json` and
`data/overlap_report.json` document the splits.

## Citation

If you use this code or data, please cite the article (citation details will be updated on
publication) and the archive: Nguyen, S. T., Huynh, T.-T., Le, H. L., and Le, T.-A. (2026).
X-AlignFact v2: code, data, logs and predictions. Zenodo. https://doi.org/10.5281/zenodo.23850412
Please also cite FEVER, NLI-FEVER and ViFactCheck (see `LICENSES.md`).
