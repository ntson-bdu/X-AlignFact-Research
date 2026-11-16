"""Part 2 - X-AlignFact architecture.

  z = LayerNorm( W_proj [ h_CLS ; sum_t alpha_t h_t ] ),  alpha = softmax(w^T H) over non-pad tokens
  logits_l = Head_l(z),  Head_l(z) = W_out Dropout(tanh(W_dense z + b)) + b_out,  l in {EN, VI}

Changes vs. v1:
  * L_proto_cross removed (it had no gradient).
  * Prototype loss is computed against the prototypes *before* the EMA update
    with the current batch.
  * Optional shared head (ablation) and optional InfoNCE (ablation).
  * The counterfactual XAI module is removed.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config
from .losses import consistency_loss, infonce_loss, prototype_loss, representation_loss

EN, VI = 0, 1


class SharedEncoder(nn.Module):
    def __init__(self, model_name: str, vocab_size: Optional[int] = None,
                 encoder_module: Optional[nn.Module] = None):
        super().__init__()
        if encoder_module is not None:          # used by the smoke test (tiny random model)
            self.encoder = encoder_module
        else:
            from transformers import AutoModel

            self.encoder = AutoModel.from_pretrained(model_name)
        if vocab_size and vocab_size > self.encoder.get_input_embeddings().num_embeddings:
            self.encoder.resize_token_embeddings(vocab_size)
        d = self.encoder.config.hidden_size
        self.hidden_dim = d
        self.attn_w = nn.Linear(d, 1)
        self.proj = nn.Linear(2 * d, d)
        self.norm = nn.LayerNorm(d)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                inputs_embeds: Optional[torch.Tensor] = None):
        if inputs_embeds is not None:          # used by Integrated Gradients (explain.py)
            H = self.encoder(inputs_embeds=inputs_embeds, attention_mask=attention_mask).last_hidden_state
        else:
            H = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        cls = H[:, 0]
        scores = self.attn_w(H).squeeze(-1).float()
        scores = scores.masked_fill(attention_mask == 0, float("-inf"))
        alpha = torch.softmax(scores, dim=-1)
        pooled = torch.bmm(alpha.unsqueeze(1).to(H.dtype), H).squeeze(1)
        z = self.norm(self.proj(torch.cat([cls, pooled], dim=-1)))
        return z, alpha


class ClassificationHead(nn.Module):
    def __init__(self, d: int, num_labels: int, dropout: float):
        super().__init__()
        self.dense = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(d, num_labels)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.out(self.drop(torch.tanh(self.dense(z))))


class EMAPrototypes(nn.Module):
    """Per-language EMA class centroids, mu_{l,k} <- g mu_{l,k} + (1-g) mean_batch(z)."""

    def __init__(self, num_labels: int, d: int, decay: float):
        super().__init__()
        self.decay = decay
        self.register_buffer("protos", torch.zeros(2, num_labels, d))
        self.register_buffer("initialised", torch.zeros(2, num_labels, dtype=torch.bool))

    def ready(self, lang: int) -> bool:
        return bool(self.initialised[lang].all())

    @torch.no_grad()
    def update(self, z: torch.Tensor, labels: torch.Tensor, lang: int) -> None:
        z = z.detach().float()
        for c in labels.unique().tolist():
            mu = z[labels == c].mean(0)
            if not self.initialised[lang, c]:
                self.protos[lang, c] = mu
                self.initialised[lang, c] = True
            else:
                self.protos[lang, c].mul_(self.decay).add_((1 - self.decay) * mu)


class XAlignFact(nn.Module):
    def __init__(self, cfg: Config, vocab_size: Optional[int] = None,
                 encoder_module: Optional[nn.Module] = None):
        super().__init__()
        self.cfg = cfg
        self.encoder = SharedEncoder(cfg.encoder, vocab_size, encoder_module)
        d = self.encoder.hidden_dim
        if cfg.shared_head:
            self.head_shared = ClassificationHead(d, cfg.num_labels, cfg.head_dropout)
        else:
            self.head_en = ClassificationHead(d, cfg.num_labels, cfg.head_dropout)
            self.head_vi = ClassificationHead(d, cfg.num_labels, cfg.head_dropout)
        self.prototypes = EMAPrototypes(cfg.num_labels, d, cfg.ema_decay)

    # ---- inference -----------------------------------------------------
    def head(self, lang: int) -> nn.Module:
        if self.cfg.shared_head:
            return self.head_shared
        return self.head_en if lang == EN else self.head_vi

    def forward(self, input_ids, attention_mask, lang: int,
                inputs_embeds: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        z, alpha = self.encoder(input_ids, attention_mask, inputs_embeds)
        return {"logits": self.head(lang)(z), "z": z, "alpha": alpha}

    # ---- training ------------------------------------------------------
    def training_losses(self, batch: Dict[str, torch.Tensor], n_streams: int
                        ) -> Dict[str, torch.Tensor]:
        """One encoder pass over all sequences of a mixed batch.

        batch keys (each optional): en_ids/en_mask/en_y, vi_ids/vi_mask/vi_y,
        par_en_ids/par_en_mask/par_vi_ids/par_vi_mask/par_y.

        Classification loss is pooled over all labelled examples and scaled by
        the number of labelled streams, so the per-step sampling proportions
        set each language's share of the gradient (the curriculum), and an
        equal EN/VI composition reproduces L_EN + L_VI exactly.
        """
        cfg = self.cfg
        parts, segs = [], []
        for key in ("en", "vi", "par_en", "par_vi"):
            if f"{key}_ids" in batch and batch[f"{key}_ids"].numel() > 0:
                parts.append((batch[f"{key}_ids"], batch[f"{key}_mask"]))
                segs.append((key, batch[f"{key}_ids"].size(0)))
        ids = torch.cat([p[0] for p in parts]).long()
        mask = torch.cat([p[1] for p in parts]).long()
        z_all, _ = self.encoder(ids, mask)
        z = dict(zip([s[0] for s in segs], torch.split(z_all, [s[1] for s in segs])))

        losses: Dict[str, torch.Tensor] = {}
        ce_sum, n_lab = z_all.new_zeros((), dtype=torch.float32), 0
        logits_par = {}
        for key, lang in (("en", EN), ("vi", VI)):
            if key in z:
                lg = self.head(lang)(z[key]).float()
                ce_sum = ce_sum + F.cross_entropy(lg, batch[f"{key}_y"], reduction="sum")
                n_lab += lg.size(0)
        if "par_en" in z:
            logits_par["en"] = self.head(EN)(z["par_en"]).float()
            logits_par["vi"] = self.head(VI)(z["par_vi"]).float()
            if cfg.parallel_ce:
                for k in ("en", "vi"):
                    ce_sum = ce_sum + F.cross_entropy(logits_par[k], batch["par_y"], reduction="sum")
                    n_lab += logits_par[k].size(0)
        if n_lab > 0:
            losses["L_task"] = n_streams * ce_sum / n_lab

        if "par_en" in z:
            if cfg.w_cons > 0:
                losses["L_cons"] = consistency_loss(logits_par["en"], logits_par["vi"])
            if cfg.w_repr > 0:
                losses["L_repr"] = representation_loss(z["par_en"], z["par_vi"])
            if cfg.w_infonce > 0:
                losses["L_infonce"] = infonce_loss(
                    z["par_en"], z["par_vi"], cfg.temperature_infonce,
                    batch["par_y"], cfg.infonce_mask_same_label)

        if cfg.w_proto > 0:
            terms = []
            for key, lang in (("en", EN), ("vi", VI)):
                if key in z:
                    y = batch[f"{key}_y"]
                    if self.prototypes.ready(lang):   # loss against prototypes before update
                        terms.append(prototype_loss(z[key], y, self.prototypes.protos[lang].clone(),
                                                    cfg.temperature_proto))
                    self.prototypes.update(z[key], y, lang)
            if terms:
                losses["L_proto"] = torch.stack(terms).mean()

        weights = {"L_task": 1.0, "L_cons": cfg.w_cons, "L_repr": cfg.w_repr,
                   "L_proto": cfg.w_proto, "L_infonce": cfg.w_infonce}
        losses["total"] = sum(weights[k] * v for k, v in losses.items())
        return losses
