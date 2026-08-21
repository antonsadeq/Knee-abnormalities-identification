"""The masking is the part that fails silently.

An unmasked softmax over slots still trains, still converges, and quietly spends most of its attention
on zero-filled slots that were never acquired — `Axial_0` exists in only 19.5 % of studies. The
decisive test is invariance: perturbing an absent slot must not move the output *at all*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from rsna_knee.config import LABELS  # noqa: E402
from rsna_knee.model import (  # noqa: E402
    N_SLOTS,
    AttentionPool,
    DiagnosisAttention,
    weighted_bce,
)


def _slots(b=2, dim=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(b, N_SLOTS, dim, generator=g)


# --- slot masking -------------------------------------------------------------------------------

def test_absent_slots_cannot_influence_the_output():
    torch.manual_seed(0)
    head = DiagnosisAttention(32, len(LABELS)).eval()
    slots = _slots()
    present = torch.tensor([[True, True, False, True, False, True]] * 2)

    with torch.no_grad():
        before = head(slots, present)
        poisoned = slots.clone()
        poisoned[:, 2] = 1e3      # absent slot, deliberately extreme
        poisoned[:, 4] = -1e3
        after = head(poisoned, present)

    assert torch.allclose(before, after, atol=1e-5)


def test_attention_renormalises_over_present_slots_only():
    torch.manual_seed(0)
    head = DiagnosisAttention(16, len(LABELS)).eval()
    slots = _slots(b=1, dim=16)
    present = torch.tensor([[True, False, True, False, False, False]])

    keys = head.key(slots)
    scores = torch.einsum("ld,bsd->bls", head.query, keys) * head.scale
    scores = scores.masked_fill(~present.unsqueeze(1), float("-inf"))
    attention = scores.softmax(dim=-1)

    assert torch.allclose(attention.sum(-1), torch.ones(1, len(LABELS)), atol=1e-6)
    assert attention[0, :, 1].abs().max() == 0
    assert attention[0, :, 3].abs().max() == 0


def test_a_study_with_no_present_slots_gives_finite_output():
    torch.manual_seed(0)
    head = DiagnosisAttention(16, len(LABELS)).eval()
    present = torch.zeros(1, N_SLOTS, dtype=torch.bool)
    with torch.no_grad():
        out = head(_slots(b=1, dim=16), present)
    assert torch.isfinite(out).all()


def test_each_diagnosis_gets_its_own_attention_pattern():
    torch.manual_seed(0)
    head = DiagnosisAttention(32, len(LABELS)).eval()
    slots = _slots(b=1)
    present = torch.ones(1, N_SLOTS, dtype=torch.bool)

    keys = head.key(slots)
    scores = torch.einsum("ld,bsd->bls", head.query, keys) * head.scale
    attention = scores.softmax(dim=-1)[0]
    # Twelve queries should not collapse onto one distribution at init.
    assert attention.std(dim=0).max() > 1e-3


def test_output_shape():
    head = DiagnosisAttention(32, len(LABELS)).eval()
    out = head(_slots(b=3), torch.ones(3, N_SLOTS, dtype=torch.bool))
    assert out.shape == (3, len(LABELS))


# --- slice pooling ------------------------------------------------------------------------------

def test_slice_pool_weights_sum_to_one_and_respect_the_mask():
    torch.manual_seed(0)
    pool = AttentionPool(16).eval()
    x = torch.randn(2, N_SLOTS, 5, 16)
    out = pool(x)
    assert out.shape == (2, N_SLOTS, 16)

    mask = torch.ones(2, N_SLOTS, 5, dtype=torch.bool)
    mask[..., 3:] = False
    with torch.no_grad():
        masked = pool(x, mask)
        truncated = pool(x[..., :3, :])
    assert torch.allclose(masked, truncated, atol=1e-6)


def test_slice_pool_is_not_a_plain_mean():
    torch.manual_seed(1)
    pool = AttentionPool(8).eval()
    x = torch.randn(1, 1, 6, 8)
    with torch.no_grad():
        assert not torch.allclose(pool(x), x.mean(dim=-2), atol=1e-3)


# --- loss ---------------------------------------------------------------------------------------

def test_zero_weight_cells_contribute_nothing():
    logits = torch.randn(4, len(LABELS))
    targets = torch.rand(4, len(LABELS))
    weights = torch.ones(4, len(LABELS))
    weights[:, 3] = 0.0

    full = weighted_bce(logits, targets, weights)
    # Corrupting a zero-weight cell's target must not change the loss.
    other = targets.clone()
    other[:, 3] = 1.0 - other[:, 3]
    assert torch.allclose(full, weighted_bce(logits, other, weights))


def test_loss_scale_is_independent_of_how_confident_the_batch_is():
    torch.manual_seed(0)
    logits = torch.randn(8, len(LABELS))
    targets = torch.rand(8, len(LABELS))
    # Same relative weights, ten times smaller: normalising by the weight sum keeps the scale.
    a = weighted_bce(logits, targets, torch.full_like(targets, 1.0))
    b = weighted_bce(logits, targets, torch.full_like(targets, 0.1))
    assert torch.allclose(a, b, atol=1e-6)


def test_loss_rewards_matching_a_graded_target():
    targets = torch.full((1, len(LABELS)), 0.8)
    weights = torch.ones(1, len(LABELS))
    close = weighted_bce(torch.full((1, len(LABELS)), 1.4), targets, weights)
    far = weighted_bce(torch.full((1, len(LABELS)), -1.4), targets, weights)
    assert close < far


def test_gold_upweighting_dominates_silence():
    torch.manual_seed(0)
    logits = torch.zeros(2, len(LABELS))
    targets = torch.tensor([[1.0] * len(LABELS), [0.0] * len(LABELS)])
    # Row 0 is a gold study at weight 8, row 1 is silence at 0.15.
    weights = torch.tensor([[8.0] * len(LABELS), [0.15] * len(LABELS)])
    loss = weighted_bce(logits, targets, weights)
    # Pushing logits toward the gold row's target must help more than toward the silent row's.
    toward_gold = weighted_bce(torch.full_like(logits, 1.0), targets, weights)
    toward_silence = weighted_bce(torch.full_like(logits, -1.0), targets, weights)
    assert toward_gold < loss < toward_silence


# --- information the preprocessing established and the aggregation must not discard ---------------

from rsna_knee.model import KneeModel, ModelConfig  # noqa: E402


def _tiny(**kw):
    """A real KneeModel small enough to run on CPU in a test.

    `DiagnosisAttention.out` is zero-initialised in the real model so training starts from logit 0
    (probability 0.5) — deliberate, and it costs one step of encoder gradient. But it also means the
    model returns exactly zero for every input at init, through which no test could observe anything,
    so it is randomised here.
    """
    config = ModelConfig(backbone="resnet18", pretrained=False, dim=32, max_slices=16, **kw)
    model = KneeModel(config).eval()
    with torch.no_grad():
        model.diagnosis.out.copy_(torch.randn_like(model.diagnosis.out) * 0.5)
    return model


def test_slot_embedding_distinguishes_otherwise_identical_slots():
    torch.manual_seed(0)
    model = _tiny(slot_embedding=True)
    # Give the table distinct values so the effect is observable rather than zero-initialised.
    with torch.no_grad():
        model.slot_embedding.copy_(torch.randn_like(model.slot_embedding))

    pixels = torch.rand(1, N_SLOTS, 3, 64, 64)
    present = torch.ones(1, N_SLOTS, dtype=torch.bool)
    with torch.no_grad():
        base = model(pixels, present)
        # Swapping two slots' contents must change the answer once slots are identifiable.
        swapped = pixels.clone()
        swapped[:, 0], swapped[:, 4] = pixels[:, 4], pixels[:, 0]
        moved = model(swapped, present)
    assert not torch.allclose(base, moved, atol=1e-4)


def test_slice_position_makes_the_pool_order_aware():
    torch.manual_seed(0)
    model = _tiny(slice_position=True)
    with torch.no_grad():
        model.slice_position.copy_(torch.randn_like(model.slice_position))

    pixels = torch.rand(1, N_SLOTS, 4, 64, 64)
    present = torch.ones(1, N_SLOTS, dtype=torch.bool)
    with torch.no_grad():
        forward = model(pixels, present)
        reversed_stack = model(pixels.flip(dims=[2]), present)
    # `dicom.order_slices` sorts medial-to-lateral deliberately; reversing must be observable.
    assert not torch.allclose(forward, reversed_stack, atol=1e-4)


def test_without_slice_position_the_pool_is_permutation_invariant():
    torch.manual_seed(0)
    model = _tiny(slice_position=False, slot_embedding=False)
    pixels = torch.rand(1, N_SLOTS, 4, 64, 64)
    present = torch.ones(1, N_SLOTS, dtype=torch.bool)
    with torch.no_grad():
        a = model(pixels, present)
        b = model(pixels.flip(dims=[2]), present)
    assert torch.allclose(a, b, atol=1e-4)


def test_slice_position_interpolates_so_slice_count_can_change():
    model = _tiny(slice_position=True)
    pixels8 = torch.rand(1, N_SLOTS, 8, 64, 64)
    pixels4 = torch.rand(1, N_SLOTS, 4, 64, 64)
    present = torch.ones(1, N_SLOTS, dtype=torch.bool)
    with torch.no_grad():
        assert model(pixels8, present).shape == (1, len(LABELS))
        assert model(pixels4, present).shape == (1, len(LABELS))


def test_too_many_slices_is_an_error_not_a_silent_truncation():
    model = _tiny(slice_position=True)
    with pytest.raises(ValueError, match="exceeds max_slices"):
        model(torch.rand(1, N_SLOTS, 17, 32, 32), torch.ones(1, N_SLOTS, dtype=torch.bool))


def test_slot_dropout_only_acts_in_training_and_never_empties_a_study():
    torch.manual_seed(0)
    model = _tiny(slot_dropout=0.9)
    pixels = torch.rand(4, N_SLOTS, 2, 64, 64)
    present = torch.zeros(4, N_SLOTS, dtype=torch.bool)
    present[:, 0] = True                      # exactly one present slot

    model.eval()
    with torch.no_grad():
        a = model(pixels, present)
        b = model(pixels, present)
    assert torch.allclose(a, b)               # inactive at eval

    model.train()
    for _ in range(5):
        out = model(pixels, present)
        # The last surviving slot is never dropped, so the output stays finite rather than becoming
        # a uniform-attention fallback over nothing.
        assert torch.isfinite(out).all()


def test_drop_path_is_passed_through_to_the_encoder():
    model = _tiny(drop_path=0.2)
    rates = [m.drop_prob for m in model.encoder.modules() if hasattr(m, "drop_prob")]
    assert rates and max(rates) > 0


# --- aligning the loss with the metric ----------------------------------------------------------

def test_class_balance_equalises_gradient_mass():
    from rsna_knee.model import class_balance

    # Mirrors the real corpus: MCL near 1.0, Synovitis near 0.37.
    weights = pd.DataFrame({lab: [0.98] * 4 for lab in LABELS})
    weights["Synovitis"] = 0.366
    weights["Fracture"] = 0.450

    scale = class_balance(weights)
    order = {lab: i for i, lab in enumerate(LABELS)}
    # The starved classes get boosted, the confident ones trimmed.
    assert scale[order["Synovitis"]] > scale[order["MCL"]]
    assert scale[order["Fracture"]] > scale[order["MCL"]]
    # ...and the mean multiplier is 1, so switching it on does not move the effective LR.
    assert float(np.mean(scale)) == pytest.approx(1.0, abs=1e-6)


def test_class_balance_survives_an_all_zero_column():
    from rsna_knee.model import class_balance

    weights = pd.DataFrame({lab: [1.0] * 3 for lab in LABELS})
    weights["Fracture"] = 0.0
    scale = class_balance(weights)
    assert np.isfinite(scale).all()


def test_class_weight_changes_which_column_dominates_the_loss():
    torch.manual_seed(0)
    logits = torch.zeros(4, len(LABELS), requires_grad=True)
    targets = torch.full((4, len(LABELS)), 0.9)
    weights = torch.ones(4, len(LABELS))
    weights[:, 8] = 0.1                        # a starved column

    weighted_bce(logits, targets, weights).backward()
    plain = logits.grad[:, 8].abs().sum().item()

    logits.grad = None
    boost = torch.ones(len(LABELS)); boost[8] = 10.0
    weighted_bce(logits, targets, weights, boost).backward()
    balanced = logits.grad[:, 8].abs().sum().item()

    assert balanced > plain * 5


# --- EMA ----------------------------------------------------------------------------------------

def test_ema_lags_the_live_weights_and_tracks_them():
    torch.manual_seed(0)
    from rsna_knee.model import WeightEma

    model = torch.nn.Linear(4, 2)
    start = model.weight.detach().clone()
    ema = WeightEma(model, decay=0.9)

    with torch.no_grad():
        model.weight.add_(1.0)
    ema.update(model)

    shadow = ema.state_dict(model)["weight"]
    # One step at decay 0.9 moves a tenth of the way.
    assert torch.allclose(shadow, start + 0.1, atol=1e-5)
    for _ in range(200):
        ema.update(model)
    assert torch.allclose(ema.state_dict(model)["weight"], model.weight, atol=1e-3)


def test_ema_state_dict_matches_the_reference_dtypes():
    from rsna_knee.model import WeightEma

    model = torch.nn.BatchNorm1d(3)
    ema = WeightEma(model, decay=0.9)
    state = ema.state_dict(model)
    for key, value in model.state_dict().items():
        assert state[key].dtype == value.dtype, key


# --- mirror TTA ---------------------------------------------------------------------------------

def test_mirror_tta_averages_two_views_and_realigns_the_labels():
    torch.manual_seed(0)
    from rsna_knee.data import label_swap_index
    from rsna_knee.model import predict_with_mirror_tta

    model = _tiny()
    pixels = torch.rand(2, N_SLOTS, 4, 64, 64)
    present = torch.ones(2, N_SLOTS, dtype=torch.bool)
    swap = torch.as_tensor(label_swap_index())

    with torch.no_grad():
        tta = predict_with_mirror_tta(model, pixels, present, swap)
        direct = model(pixels, present)
    assert tta.shape == direct.shape
    # A real second view, so it must differ from the single pass.
    assert not torch.allclose(tta, direct, atol=1e-5)


def test_mirror_tta_is_symmetric_under_mirroring_the_input():
    """Feeding the mirror image must give the mirrored answer — the two views just swap roles."""
    torch.manual_seed(0)
    from rsna_knee.data import IN_PLANE_SLOTS, THROUGH_PLANE_SLOTS, label_swap_index
    from rsna_knee.model import predict_with_mirror_tta

    model = _tiny()
    pixels = torch.rand(1, N_SLOTS, 4, 64, 64)
    present = torch.ones(1, N_SLOTS, dtype=torch.bool)
    swap = torch.as_tensor(label_swap_index())

    flipped = pixels.clone()
    for i in IN_PLANE_SLOTS:
        flipped[:, i] = torch.flip(flipped[:, i], dims=[-1])
    for i in THROUGH_PLANE_SLOTS:
        flipped[:, i] = torch.flip(flipped[:, i], dims=[1])

    with torch.no_grad():
        a = predict_with_mirror_tta(model, pixels, present, swap)
        b = predict_with_mirror_tta(model, flipped, present, swap)
    assert torch.allclose(a, b[:, swap], atol=1e-4)
