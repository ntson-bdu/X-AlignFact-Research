"""X-AlignFact v2: reproducible re-implementation for the OJCS resubmission.

Part map (mirrors the original notebook so reviewers can follow the changes):
    Part 0  config.py, utils.py            configuration, seeding, environment capture
    Part 1  data.py, translate.py           leakage-free splits, parallel corpus, tokenization
    Part 2  model.py, losses.py             architecture (L_proto_cross removed)
    Part 3  sampler.py, train.py            per-step curriculum sampler, single training pipeline
    Part 4  evaluate.py                     test-set evaluation, CLCS family of metrics
    Part 5  variants (config.py), stats.py  ablations/baselines over seeds, uncertainty, tests
    Part 6  analysis.py, retrieval.py       t-SNE, errors, gradients, retrieved-evidence setting
"""

__version__ = "2.0.0"
