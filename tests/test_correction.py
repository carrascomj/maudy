"""Test conserved moieities functionality related to correcting."""

from random import sample
from copy import deepcopy

import torch
import pytest
from maud.data_model.maud_input import MaudInput
from maudy.analysis import predict
from maudy.train import train


def pick_two_bal_mets(mi: MaudInput) -> list[str]:
    """Pick two balanced metabolites at random from the kinetic model."""
    bal_mics: list[str] = [met.id for met in mi.kinetic_model.mics if met.balanced]
    return sample(bal_mics, k=2)


def test_sample_with_hardcoded_correction_groups(ci_aord_model: MaudInput):
    model, _ = train(ci_aord_model, 4, True, True, True, True, 1, False)
    assert len(model.correct_groups) > 0


def test_sample_with_computed_correction_groups(ci_aord_model: MaudInput):
    maud_input = deepcopy(ci_aord_model)
    hardcoded_correcting_groups = maud_input._maudy_config.correction_groups
    maud_input._maudy_config.correction_groups = []
    model, _ = train(maud_input, 4, True, True, True, True, 1, False)
    assert len(model.correct_groups) > 0
    assert len(model.correction_groups_named) == len(model.correct_groups)
    assert all(
        [
            any(
                set(correct_group) == set(expected_group)
                for expected_group in hardcoded_correcting_groups
            )
            for correct_group in model.correction_groups_named
        ]
    ), f"Computed correction groups ({model.correction_groups_named}) do not match the expected ones."


@pytest.mark.parametrize("epochs", range(2,6))
def test_toy_models_with_random_correction_do_not_raise_nan(maud_input: MaudInput, epochs: int):
    maud_input = deepcopy(maud_input)
    maud_input._maudy_config.correction_groups = [pick_two_bal_mets(maud_input)]
    with torch.autograd.detect_anomaly(True):
        model, _ = train(maud_input, epochs, True, True, True, True, 1, True)
    assert len(model.correct_groups) == 1


def test_ci_aord_do_not_raise_nan(ci_aord_model: MaudInput):
    with torch.autograd.detect_anomaly(True):
        model, _ = train(ci_aord_model, 4, True, True, True, True, 1, True)
    assert len(model.correct_groups) == 2


def test_correction_group_concentration_is_maintained(maud_input: MaudInput):
    maud_input = deepcopy(maud_input)
    maud_input._maudy_config.correction_groups = [pick_two_bal_mets(maud_input)]
    model, _ = train(maud_input, 10, True, True, True, True, 1, False)
    samples = predict(model, 100, var_names=("ln_bal_conc", "correction"))
    i = 0
    for group_idx in model.correct_groups:
        i += 1
        x = samples["ln_bal_conc"][:, :, group_idx]
        q = samples["correction"][:, :, group_idx]
        delta = abs(x.exp().sum(dim=-1) - (x - q).exp().sum(dim=-1))
        hits = delta <= x.exp().sum(dim=-1) * 0.001
        assert hits.all(), f"Unequal quantities, max delta = {delta.max()} where x={x[hits]}; q={q[hits]}"
    assert len(model.correct_groups) > 0
