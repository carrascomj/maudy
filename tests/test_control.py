"""Test for metabolic control."""

import pytest
import torch
from maudy.model import Maudy
from maudy.control import control_matrices, get_jacobian, mca, inverse_function
from maudy.train import train
from maud.loading_maud_inputs import MaudInput
from xarray import DataArray


def test_inverse_function_recovers_snapshot(methionine_model: MaudInput, elasticities: DataArray, concentration_control_matrix: DataArray):
    elas = torch.from_numpy(elasticities.to_numpy()).double()
    model = Maudy(methionine_model)
    N = model.S[model.balanced_mics_idx, :].double()
    c_s, _ = inverse_function(elas, N)
    ccm = torch.from_numpy(concentration_control_matrix.to_numpy()).double()
    assert torch.allclose(c_s, ccm, atol=1e-6, rtol=1e-5)


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
