"""Test conserved moieities functionality related to quenching."""

import torch
import pytest
from maud.data_model.maud_input import MaudInput
from maudy.train import train


@pytest.mark.parametrize("epochs", range(2,6))
def test_toy_models_with_quenching_do_not_raise_nan(maud_input: MaudInput, epochs: int):
    with torch.autograd.detect_anomaly(True):
        model, _ = train(maud_input, epochs, True, True, True, True, 10, True)
    assert model.should_quench


def test_ci_aord_do_not_raise_nan(ci_aord_model: MaudInput):
    with torch.autograd.detect_anomaly(True):
        model, _ = train(ci_aord_model, 4, True, True, True, True, 10, True)
    assert model.should_quench
