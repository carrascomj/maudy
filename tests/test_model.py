"""Test model is wellformed upon loading."""

import warnings

import pytest
from copy import deepcopy

from maud.data_model.kinetic_model import ReactionMechanism
from maudy.model import Maudy


@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("quench", [True, False])
def test_load_product_does_not_raise(maud_input, normalize, quench):
    Maudy(maud_input=maud_input, normalize=normalize, quench=quench)


def test_all_reversible_mechanisms_are_correct(maud_input):
    maudy = Maudy(maud_input)
    assert sum(maudy.irreversible) == sum(
        reac.mechanism == ReactionMechanism.reversible_michaelis_menten
        for reac in maud_input.kinetic_model.reactions
    )
    assert sum(~maudy.irreversible) == sum(
        reac.mechanism == ReactionMechanism.irreversible_michaelis_menten
        for reac in maud_input.kinetic_model.reactions
    )


@pytest.mark.parametrize(
    "temperature,dgf_water,should_warn",
    [
        (310.15, -165.20171502346653, False),
        (298.15, 150.9, False),
        (298.15, 160.9, True),
    ],
)
def test_temp_dgf_pairs_do_not_raise_warnings(
    maud_input, temperature, dgf_water, should_warn
):
    maud_input = deepcopy(maud_input)
    maud_input._maudy_config.temperature = temperature
    maud_input._maudy_config.dgf_water = dgf_water

    with warnings.catch_warnings(record=True) as warning_list:
        model = Maudy(maud_input=maud_input)

    assert model.temperature == temperature
    assert model.dgf_water == dgf_water
    unwanted_warning = "not seem to match the approximate relationship"
    assert (
        any(
            unwanted_warning in str(warning.message)
            for warning in warning_list
            if isinstance(warning.message, UserWarning)
        )
        == should_warn
    )
