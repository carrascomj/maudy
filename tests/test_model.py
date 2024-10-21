"""Test model is wellformed upon loading."""

import pytest

from maud.data_model.kinetic_model import ReactionMechanism
from maudy.model import Maudy


@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("quench", [True, False])
def test_load_product_does_not_raise(maud_input, normalize, quench):
    Maudy(maud_input=maud_input, normalize=normalize, quench=quench)


def test_all_reversible_mechanisms_are_correct(maud_input):
    maudy = Maudy(maud_input)
    assert (
        sum(maudy.irreversible) == 
        sum(1 for reac in maud_input.kinetic_model.reactions if reac.mechanism == ReactionMechanism.reversible_michaelis_menten)
    )
    assert (
        sum(~maudy.irreversible) == 
        sum(1 for reac in maud_input.kinetic_model.reactions if reac.mechanism == ReactionMechanism.irreversible_michaelis_menten)
    )
