"""Test conserved moieities functionality related to quenching."""

from random import sample
from copy import deepcopy

import torch
import pytest
from maud.data_model.maud_input import MaudInput
from maudy.analysis import predict
from maudy.train import train


@pytest.mark.parametrize("epochs", range(2,6))
def test_toy_models_with_quenching_do_not_raise_nan(maud_input: MaudInput, epochs: int):
    with torch.autograd.detect_anomaly(True):
        _ = train(maud_input, epochs, True, True, True, True, 10, True)


def test_ci_aord_do_not_raise_nan(ci_aord_model: MaudInput):
    with torch.autograd.detect_anomaly(True):
        _ = train(ci_aord_model, 4, True, True, True, True, 10, True)
