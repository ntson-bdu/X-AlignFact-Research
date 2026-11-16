"""Part 2 - Loss functions. All are computed in float32.

Formulas (to be copied verbatim into the manuscript):
  L_cons  = 1/2 [ KL(p_EN || p_VI) + KL(p_VI || p_EN) ]          (batch mean)
  L_repr  = 1/(B d) sum_i || z_i^EN - z_i^VI ||_2^2                (mean squared error)
  L_proto = -1/N sum_i log softmax_k( cos(z_i, mu_{l_i,k}) / tau )[y_i]
  L_NCE   = symmetric cross-lingual InfoNCE over the parallel pairs of a batch
            (ablation only; not part of the proposed model)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def consistency_loss(logits_en: torch.Tensor, logits_vi: torch.Tensor) -> torch.Tensor:
    le, lv = logits_en.float(), logits_vi.float()
    log_pe, log_pv = F.log_softmax(le, -1), F.log_softmax(lv, -1)
    kl_en_vi = F.kl_div(log_pv, log_pe, log_target=True, reduction="batchmean")  # KL(p_EN || p_VI)
    kl_vi_en = F.kl_div(log_pe, log_pv, log_target=True, reduction="batchmean")  # KL(p_VI || p_EN)
    return 0.5 * (kl_en_vi + kl_vi_en)


def representation_loss(z_en: torch.Tensor, z_vi: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(z_en.float(), z_vi.float())


def prototype_loss(z: torch.Tensor, labels: torch.Tensor, protos: torch.Tensor,
                   tau: float) -> torch.Tensor:
    """z: (N, d) with gradient; protos: (C, d) EMA buffer (no gradient)."""
    z_n = F.normalize(z.float(), dim=-1)
    p_n = F.normalize(protos.float(), dim=-1)
    return F.cross_entropy(z_n @ p_n.T / tau, labels)


def infonce_loss(z_en: torch.Tensor, z_vi: torch.Tensor, tau: float,
                 labels: torch.Tensor | None = None,
                 mask_same_label: bool = False) -> torch.Tensor:
    """Positives: (z_en_i, z_vi_i). Negatives: other pairs in the batch.

    With mask_same_label=True, negatives that share the label of the anchor are
    removed (tests the hypothesis that InfoNCE conflicts with classification by
    pushing apart same-label claims).
    """
    a = F.normalize(z_en.float(), dim=-1)
    b = F.normalize(z_vi.float(), dim=-1)
    sim = a @ b.T / tau
    if mask_same_label and labels is not None:
        same = labels.unsqueeze(0) == labels.unsqueeze(1)
        off_diag = ~torch.eye(len(labels), dtype=torch.bool, device=sim.device)
        sim = sim.masked_fill(same & off_diag, float("-inf"))
    target = torch.arange(sim.size(0), device=sim.device)
    return 0.5 * (F.cross_entropy(sim, target) + F.cross_entropy(sim.T, target))
