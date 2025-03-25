"""Test for metabolic control."""

import pytest
import torch
from maudy.control import control_matrices, get_jacobian, mca
from maudy.train import train


@pytest.mark.parametrize("prior_str", ["unb_conc", "enzyme_conc", "kcat", "km"])
def test_jacobian_shape_matches_expected_conc(maud_input, prior_str: str):
    model, _ = train(maud_input, 4, True, False, True, True, 1, False)
    samples = 9
    jacobian = get_jacobian(model, prior_wrt=prior_str, samples=samples)
    assert (~torch.isnan(jacobian)).all()
    if prior_str not in ["kcat", "km"]:
        assert jacobian.shape[1] == len(model.experiments)
        assert len(jacobian.shape) == 4
    assert jacobian.shape[-2] == model.bal_conc_mu.shape[-1]
    assert samples == jacobian.shape[0], "first dim must be number of samples"


@pytest.mark.parametrize("prior_str", ["unb_conc", "enzyme_conc", "kcat", "km"])
def test_jacobian_shape_matches_expected_flux(maud_input, prior_str: str):
    model, _ = train(maud_input, 4, True, False, True, True, 1, False)
    samples = 9
    jacobian = get_jacobian(model, prior_wrt=prior_str, d_conc=False, samples=samples)
    assert (~torch.isnan(jacobian)).all()
    if prior_str not in ["kcat", "km"]:
        assert jacobian.shape[1] == len(model.experiments)
        assert len(jacobian.shape) == 4
    assert jacobian.shape[-2] == model.num_reactions
    assert samples == jacobian.shape[0], "first dim must be number of samples"


def test_control_matrices_shapes_match_expectations(maud_input):
    model, _ = train(maud_input, 4, True, False, True, True, 1, False)
    samples = 9
    c_s, c_j = control_matrices(model, samples=samples)
    assert c_s.shape[-2] == model.bal_conc_mu.shape[-1], "c_s must be [..., M, R]"
    assert c_s.shape[-1] == c_j.shape[-1] == c_j.shape[-2] == model.num_reactions, "c_j must be [..., R, R]; and c_s [..., R]"
    n_exp = len(model.experiments)
    assert n_exp == c_s.shape[-3]  == c_j.shape[-3], "second dim must be number of experiments"
    assert samples == c_s.shape[0], "first dim must be number of samples"


@pytest.mark.parametrize("prior_str", ["unb_conc", "enzyme_conc", "kcat", "km"])
def test_mca_shape_match_expectations(maud_input, prior_str: str):
    model, _ = train(maud_input, 4, True, False, True, True, 1, False)
    samples = 9
    matrix = mca(model, prior_wrt=prior_str, d_conc=False, samples=samples)
    assert (~torch.isnan(matrix)).all()
    if prior_str not in ["kcat", "km"]:
        assert matrix.shape[1] == len(model.experiments)
    assert len(matrix.shape) == 4
    assert matrix.shape[-2] == model.num_reactions
    assert samples == matrix.shape[0], "first dim must be number of samples"
