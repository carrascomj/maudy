"""Test conserved moieities functionality related to quenching."""

from copy import deepcopy

from maud.data_model.maud_input import MaudInput
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


def test_toy_models_with_computed_quenching_correction_groups(maud_input: MaudInput):
    maud_input = deepcopy(maud_input)
    model, _ = train(maud_input, 4, True, True, True, True, 10, False)
    # for all toy models, we expect one quenching correction group
    assert len(model.quench_groups) == 1
