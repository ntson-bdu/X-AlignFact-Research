"""Part 3 - Per-step curriculum sampler.

v1 kept a constant 32 EN : 32 VI : 16 parallel composition in every step; the
phase weights only changed the size of the English pool. Here every step
contains n_EN, n_VI, n_PAR examples drawn in proportion to the phase weights
(w_EN, w_VI, w_PAR) out of a fixed budget (default 64; a parallel pair counts
as one example), so the curriculum changes what the encoder is trained on.

Each stream draws without replacement from its own seeded permutation and
reshuffles when exhausted, so a run is reproducible from (data, seed) alone and
is not affected by evaluation code consuming random numbers.
"""
from __future__ import annotations

from typing import List

import torch


def allocate(weights: List[float], budget: int) -> List[int]:
    """Largest-remainder rounding; streams with weight > 0 get at least 1 slot."""
    raw = [w * budget for w in weights]
    counts = [int(r) for r in raw]
    for i, w in enumerate(weights):
        if w > 0 and counts[i] == 0:
            counts[i] = 1
    rem = budget - sum(counts)
    order = sorted(range(len(raw)), key=lambda i: raw[i] - int(raw[i]), reverse=True)
    i = 0
    while rem > 0:
        j = order[i % len(order)]
        if weights[j] > 0:
            counts[j] += 1
            rem -= 1
        i += 1
    while rem < 0:  # only possible after the minimum-1 rule
        j = max(range(len(counts)), key=lambda k: counts[k])
        counts[j] -= 1
        rem += 1
    return counts


class StreamIterator:
    def __init__(self, n: int, seed: int):
        self.n = n
        self.g = torch.Generator()
        self.g.manual_seed(seed)
        self.perm = torch.randperm(n, generator=self.g)
        self.pos = 0
        self.epochs_completed = 0

    def take(self, k: int) -> torch.Tensor:
        out = []
        while k > 0:
            if self.pos >= self.n:
                self.perm = torch.randperm(self.n, generator=self.g)
                self.pos = 0
                self.epochs_completed += 1
            step = min(k, self.n - self.pos)
            out.append(self.perm[self.pos:self.pos + step])
            self.pos += step
            k -= step
        return torch.cat(out) if out else torch.empty(0, dtype=torch.long)


class CurriculumSampler:
    STREAMS = ("en", "vi", "par")

    def __init__(self, cfg, sizes: dict, seed: int):
        self.cfg = cfg
        # distinct, fixed offsets so each stream has its own random sequence
        self.iters = {s: StreamIterator(sizes[s], seed * 1000 + i)
                      for i, s in enumerate(self.STREAMS) if sizes.get(s, 0) > 0}

    def counts(self, epoch: int) -> List[int]:
        """Counts from the phase weights; disabled streams are then set to 0 WITHOUT
        redistributing their slots, so e.g. `no_parallel` keeps exactly the EN/VI
        composition of `full` and only removes the parallel pairs."""
        c = allocate(self.cfg.phase_weights(epoch), self.cfg.step_budget)
        return [ci if m else 0 for ci, m in zip(c, self.cfg.stream_mask())]

    def next(self, epoch: int) -> dict:
        c = dict(zip(self.STREAMS, self.counts(epoch)))
        return {s: self.iters[s].take(c[s]) for s in self.STREAMS if c[s] > 0 and s in self.iters}
