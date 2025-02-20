"""Test for metabolic control."""

import pytest
import torch
from maudy.control import get_jacobian
from maudy.train import train


@pytest.mark.parametrize("prior_str", ["unb_conc", "enzyme_conc", "kcat", "km"])
def test_jacobian_shape_matches_expected(maud_input, prior_str: str):
    model, _ = train(maud_input, 4, True, False, True, True, 1, False)
    jacobian = get_jacobian(model, prior_wrt=prior_str)
    print(jacobian.shape)
    assert (~torch.isnan(jacobian)).all()
    if prior_str not in ["kcat", "km"]:
        assert jacobian.shape[0] == len(model.experiments)
        assert len(jacobian.shape) == 3
    assert jacobian.shape[-2] == model.bal_conc_mu.shape[-1]
