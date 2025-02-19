"""Test for metabolic control."""

import torch
from maudy.control import get_jacobian
from maudy.train import train

def test_jacobian_shape_matches_expected(maud_input):
    model, _ = train(maud_input, 20, True, False, True, True, 10, False)
    jacobian = get_jacobian(model, prior_wrt="enzyme_conc")
    assert (~torch.isnan(jacobian)).all()
    assert jacobian.shape[0] == len(model.experiments)
    assert jacobian.shape[1] == model.bal_conc_mu.shape[-1]
    assert jacobian.shape[2] == model.enzyme_concs_loc.shape[-1]
