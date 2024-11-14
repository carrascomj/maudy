"""Test conserved moieities functionality related to quenching."""

from copy import deepcopy

import torch
import pytest
from maud.data_model.maud_input import MaudInput
from maudy.analysis import predict
from maudy.train import train


def test_sample_with_hardcoded_quenching_correction_groups(ci_aord_model: MaudInput):
    model, _ = train(ci_aord_model, 4, True, True, True, True, 10, False)
    assert len(model.quench_groups) > 0


def test_sample_with_computed_correction_groups(ci_aord_model: MaudInput):
    maud_input = deepcopy(ci_aord_model)
    hardcoded_quenching_groups = maud_input._maudy_config.quenching_groups
    maud_input._maudy_config.quenching_groups = []
    model, _ = train(maud_input, 4, True, True, True, True, 10, False)
    assert len(model.quench_groups) > 0
    assert len(model.quenching_groups_named) == len(model.quench_groups)
    assert all(
        [
            any(
                set(quench_group) == set(expected_group)
                for expected_group in hardcoded_quenching_groups
            )
            for quench_group in model.quenching_groups_named
        ]
    ), f"Computed quenching groups ({model.quenching_groups_named}) do not match the expected ones."


@pytest.mark.parametrize("epochs", range(2,6))
def test_toy_models_with_computed_quenching_correction_do_not_raise_nan(maud_input: MaudInput, epochs: int):
    with torch.autograd.detect_anomaly(True):
        model, _ = train(maud_input, epochs, True, True, True, True, 10, True)
    # for all toy models, we expect one quenching correction group
    assert len(model.quench_groups) == 1


def test_ci_aord_do_not_raise_nan(maud_input: MaudInput):
    with torch.autograd.detect_anomaly(True):
        model, _ = train(maud_input, 4, True, True, True, True, 10, True)
    # for all toy models, we expect one quenching correction group
    assert len(model.quench_groups) == 1


def test_quenching_group_concentration_is_maintained(maud_input: MaudInput):
    model, _ = train(maud_input, 10, True, True, True, True, 10, False)
    samples = predict(model, 100, var_names=("ln_bal_conc", "quench_correction"))
    for group_idx in model.quench_groups:
        x = samples["ln_bal_conc"][:, :, group_idx]
        q = samples["quench_correction"][:, :, group_idx]
        assert (abs(x.exp().sum(dim=-1) - (x - q).exp().sum(dim=-1)) <= x.min().exp() * 0.001).all()
