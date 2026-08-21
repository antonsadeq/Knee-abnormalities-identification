"""Study-level multi-label model: encode slices, pool to slots, then one query per diagnosis.

The supervision is twelve binary flags attached to a **study** — not to a slice, not to a pixel and
not to a series. Nothing in the data says *where* a finding is. That single fact determines the whole
shape of this module, in two directions:

**It rules out a segmentation decoder.** A per-pixel head has no target and no use; you would train it
through a pooling layer, paying its parameters for a spatial output you immediately discard. Encoder
only.

**It caps how rich the aggregator may be.** Where supervision is coarse, so is the head. A deep
transformer over slots has nothing to learn from twelve study-level bits, so the aggregation here is
deliberately thin: attention pooling over slices, then a single masked attention over slots with one
learned query per diagnosis, then a scalar per diagnosis. Roughly 12 x d parameters of "head".

Why a query *per diagnosis* rather than a mean over slots: each finding is read on particular
sequences — an effusion on fluid-sensitive images, cartilage on others — so averaging the six slots
dilutes the one carrying the evidence with five that do not. And why the softmax is **masked**: only
566 of 4,407 studies have all six slots and `Axial_0` is present in just 19.5 %, so an unmasked
softmax would spend most of its mass on zero-filled slots and force the model to distinguish "black"
from "not acquired".
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from .config import LABELS
from .dicom import SLOTS

N_SLOTS = len(SLOTS)


@dataclass
class ModelConfig:
    """Everything that changes between the accuracy model and the efficiency model."""

    backbone: str = "convnext_tiny"
    pretrained: bool = True
    n_labels: int = len(LABELS)
    dim: int = 256
    dropout: float = 0.1
    #: Stochastic depth in the encoder. M1 overfitted from epoch 8 with none of it.
    drop_path: float = 0.0
    #: Probability of hiding a *present* slot during training. Regularises the slot attention against
    #: exactly the condition it meets at test time — most studies are missing two of the six slots.
    slot_dropout: float = 0.0
    #: Add a learned identity vector per slot. Without it the attention sees six anonymous vectors and
    #: has to infer plane and sequence from content, though the slot index states both exactly.
    slot_embedding: bool = True
    #: Add a learned position per slice before pooling. `dicom.order_slices` sorts slices medial-to-
    #: lateral on purpose; a permutation-invariant pool throws that ordering away.
    slice_position: bool = True
    #: Longest slice run the position table must cover.
    max_slices: int = 32
    #: Gradient checkpointing on the encoder. A batch of B studies pushes B x 6 x n_slices images
    #: through it, so activation memory, not parameters, is the binding constraint.
    checkpoint: bool = False


class AttentionPool(nn.Module):
    """Gated attention pooling over a sequence (Ilse et al.), with an optional mask.

    Used to collapse slices within a slot. A mean would let the many unremarkable slices outvote the
    one or two showing the tear, which is the entire difficulty of a study-level label; a max would
    key on a single noisy slice. Gated attention sits between and is cheap.
    """

    def __init__(self, dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.attend = nn.Linear(dim, hidden)
        self.gate = nn.Linear(dim, hidden)
        self.score = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: (..., n, dim) -> (..., dim)
        weights = self.score(torch.tanh(self.attend(x)) * torch.sigmoid(self.gate(x)))
        if mask is not None:
            weights = weights.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        weights = weights.softmax(dim=-2)
        return (weights * x).sum(dim=-2)


class DiagnosisAttention(nn.Module):
    """One learned query per diagnosis, attending over the slot embeddings.

    The mask is the load-bearing part. ``present`` marks which slots the study actually contains, and
    the softmax is renormalised over those alone — so a study with four slots distributes all of its
    attention across four, rather than leaking most of it onto two zero vectors.

    A study with *no* present slots would make the softmax undefined; that cannot happen in this cache
    (every study has at least three) but it is handled rather than left to produce silent NaNs.
    """

    def __init__(self, dim: int, n_labels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(n_labels, dim) * dim**-0.5)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
        self.scale = dim**-0.5
        # One scalar per diagnosis, reading only its own attended context.
        self.out = nn.Parameter(torch.zeros(n_labels, dim))
        self.bias = nn.Parameter(torch.zeros(n_labels))

    def forward(self, slots: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
        # slots: (B, n_slots, dim), present: (B, n_slots) bool -> logits (B, n_labels)
        keys = self.key(slots)
        values = self.value(slots)

        scores = torch.einsum("ld,bsd->bls", self.query, keys) * self.scale
        usable = present.unsqueeze(1)                      # (B, 1, n_slots)
        empty = ~present.any(dim=1)                        # studies with nothing present
        scores = scores.masked_fill(~usable, float("-inf"))
        # Give an empty study a uniform distribution instead of a NaN.
        scores = torch.where(empty[:, None, None], torch.zeros_like(scores), scores)

        attention = scores.softmax(dim=-1)
        context = self.drop(torch.einsum("bls,bsd->bld", attention, values))
        return (context * self.out).sum(-1) + self.bias


class KneeModel(nn.Module):
    """Slices -> slots -> twelve logits.

    Input is ``(B, n_slots, n_slices, H, W)`` in 0-1. Every slice goes through one shared encoder,
    which is what makes the parameter count independent of how many slices or slots you feed it — the
    efficiency submission can then read fewer of both without retraining a different architecture.
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        import timm

        self.config = config or ModelConfig()
        # in_chans=1 rather than repeating grey to 3 channels: timm sums the pretrained stem weights
        # across the input channels, which keeps the pretrained filters and a third of the stem FLOPs.
        encoder_kwargs = dict(
            pretrained=self.config.pretrained, in_chans=1, num_classes=0
        )
        if self.config.drop_path:
            encoder_kwargs["drop_path_rate"] = self.config.drop_path
        self.encoder = timm.create_model(self.config.backbone, **encoder_kwargs)
        if self.config.checkpoint and hasattr(self.encoder, "set_grad_checkpointing"):
            self.encoder.set_grad_checkpointing(True)

        feature_dim = self.encoder.num_features
        self.project = nn.Sequential(
            nn.Linear(feature_dim, self.config.dim),
            nn.LayerNorm(self.config.dim),
            nn.GELU(),
        )
        self.slice_pool = AttentionPool(self.config.dim)
        self.slot_norm = nn.LayerNorm(self.config.dim)

        # Both tables are tiny — 6 x dim and max_slices x dim — and both restore information the
        # preprocessing already established but the aggregation would otherwise discard.
        self.slot_embedding = (
            nn.Parameter(torch.zeros(N_SLOTS, self.config.dim))
            if self.config.slot_embedding else None
        )
        self.slice_position = (
            nn.Parameter(torch.zeros(self.config.max_slices, self.config.dim))
            if self.config.slice_position else None
        )
        self.diagnosis = DiagnosisAttention(
            self.config.dim, self.config.n_labels, self.config.dropout
        )

    def forward(self, pixels: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
        b, n_slots, n_slices, h, w = pixels.shape
        # The batch arrives as (B, slots, slices, H, W); channels_last is a rank-4 layout, so it is
        # applied here after flattening rather than by the caller on the 5-D tensor.
        flat = pixels.reshape(b * n_slots * n_slices, 1, h, w)
        if pixels.is_cuda:
            flat = flat.contiguous(memory_format=torch.channels_last)
        features = self.encoder(flat)
        features = self.project(features).reshape(b, n_slots, n_slices, -1)

        if self.slice_position is not None:
            if n_slices > self.slice_position.shape[0]:
                raise ValueError(
                    f"{n_slices} slices exceeds max_slices={self.slice_position.shape[0]}"
                )
            # Interpolated across the table rather than indexed 0..n-1, so a model trained with 8
            # slices still sees a consistent notion of "a third of the way through the stack" when
            # run with 16 — otherwise changing the slice count silently shifts every position.
            grid = torch.linspace(0, self.slice_position.shape[0] - 1, n_slices,
                                  device=features.device)
            low, high = grid.floor().long(), grid.ceil().long()
            frac = (grid - low).unsqueeze(-1)
            position = self.slice_position[low] * (1 - frac) + self.slice_position[high] * frac
            features = features + position

        slots = self.slot_norm(self.slice_pool(features))

        if self.slot_embedding is not None:
            slots = slots + self.slot_embedding

        mask = present.bool()
        if self.training and self.config.slot_dropout > 0:
            # Never drop the last surviving slot: a study with nothing present carries no signal at
            # all, which is a different and useless training example.
            keep = torch.rand(mask.shape, device=mask.device) >= self.config.slot_dropout
            candidate = mask & keep
            mask = torch.where(candidate.any(dim=1, keepdim=True), candidate, mask)

        return self.diagnosis(slots, mask)


def weighted_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    class_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-cell weighted BCE against graded targets.

    The weights carry the labeller's confidence: 1.0 where the report was explicit, ~0 where it
    hedged or the two labellers flatly disagreed, 0.15 where the report was silent, and 8.0 on the 58
    image-derived gold studies. Normalising by the weight sum rather than the cell count keeps the
    gradient scale independent of how much of the batch happens to be confident — otherwise a batch
    full of silence produces a tiny gradient and a batch of gold studies a huge one.

    ``class_weight`` corrects a mismatch between this loss and the competition metric. Confidence
    mass is very unevenly distributed across findings — mean weight runs from 0.980 on MCL down to
    0.450 on Fracture and **0.366 on Synovitis** — so without it Synovitis contributes barely a third
    of MCL's gradient. But the metric is the *unweighted* mean of twelve AUCs: every column counts the
    same, and the macro bleeds from whichever column is weakest. Scaling each class by the inverse of
    its corpus-wide mean weight equalises the gradient mass and aligns the objective with the score.

    Computed from the whole training set rather than per batch on purpose. A per-batch column
    normaliser divides by the weight mass of two studies, which is noisy enough to swamp the effect
    it is trying to correct.
    """
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    if class_weight is not None:
        weights = weights * class_weight
    total = weights.sum().clamp_min(1e-6)
    return (loss * weights).sum() / total


def class_balance(weights: "object", labels: list[str] | None = None) -> np.ndarray:
    """Per-class multiplier equalising gradient mass, normalised to leave the overall scale alone.

    ``weights`` is the training weight frame. Returns one multiplier per label, scaled so their mean
    is 1 — otherwise switching this on would silently change the effective learning rate as well.
    """
    labels = labels or LABELS
    mean = weights.reindex(columns=labels).mean().to_numpy()
    mean = np.where(mean > 1e-6, mean, 1.0)
    scale = mean.mean() / mean
    return scale / scale.mean()


class WeightEma:
    """Exponential moving average of the weights, evaluated and saved instead of the live model.

    M1's validation curve wobbled by 0.01-0.02 between adjacent late epochs while the training loss
    fell smoothly — the sign of a model bouncing around a minimum rather than sitting in it. An EMA
    costs one extra copy of the parameters and no extra compute in the backward pass, and it turns
    that wobble into a smoother curve whose peak is usually slightly higher.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        # Only float tensors are averaged in float32. Casting integer buffers — batch-norm's
        # `num_batches_tracked` is int64 — would hand `load_state_dict` the wrong dtype back.
        self.shadow = {
            k: (v.detach().clone().float() if v.dtype.is_floating_point else v.detach().clone())
            for k, v in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for key, value in model.state_dict().items():
            shadow = self.shadow[key]
            if value.dtype.is_floating_point:
                shadow.mul_(self.decay).add_(value.detach().float(), alpha=1.0 - self.decay)
            else:
                # Integer buffers (batch-norm counters) cannot be averaged; take the latest.
                self.shadow[key] = value.detach().clone()

    def state_dict(self, reference: nn.Module) -> dict:
        """EMA weights cast back to the reference model's dtypes, ready for `load_state_dict`."""
        out = {}
        for key, value in reference.state_dict().items():
            shadow = self.shadow[key]
            out[key] = shadow.to(value.dtype) if value.dtype.is_floating_point else shadow
        return out


def predict_with_mirror_tta(
    model: nn.Module, pixels: torch.Tensor, present: torch.Tensor, swap: torch.Tensor
) -> torch.Tensor:
    """Average the logits for a study and for its mirror image.

    Mirroring is the one augmentation this data supports exactly rather than approximately — knees
    come in two chiralities and `dicom.py` normalised them all to one. So the mirrored view is a
    genuine second look, not a distortion.

    The subtlety is that mirroring swaps the medial/lateral label pairs, so the mirrored pass predicts
    a *permuted* label vector. `swap` is applied to bring it back into the original ordering before
    averaging; since the permutation is an involution, the same index array does both jobs.
    """
    from .data import IN_PLANE_SLOTS, THROUGH_PLANE_SLOTS

    direct = model(pixels, present)

    flipped = pixels.clone()
    for i in IN_PLANE_SLOTS:
        flipped[:, i] = torch.flip(flipped[:, i], dims=[-1])
    for i in THROUGH_PLANE_SLOTS:
        flipped[:, i] = torch.flip(flipped[:, i], dims=[1])

    mirrored = model(flipped, present)[:, swap]
    return (direct + mirrored) / 2.0
