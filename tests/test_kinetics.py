"""Check kinetic parity with Maud."""

from typing import Optional

import torch
from maud.loading_maud_inputs import MaudInput
from maudy.model import Maudy
from maudy.kinetics import get_allostery, get_competitive_inhibition_denom, get_free_enzyme_ratio_denom, get_vmax


def format_tensor(tensor: torch.Tensor) -> str:
    """Print tensor for debugging when test fails."""
    tensor = tensor.float()
    tensor_list = tensor.tolist()
    formatted_tensor = [[f"{element:.4f}" for element in row] for row in tensor_list]

    top_bracket = "⎡ "
    middle_bracket = "⎢ "
    bottom_bracket = "⎣ "

    # build the formatted tensor with big brackets
    string_tensor = ""
    for i, row in enumerate(formatted_tensor):
        if i == 0:
            string_tensor += top_bracket + "  ".join(row) + " ⎤" + "\n"
        elif i == len(formatted_tensor) - 1:
            string_tensor += bottom_bracket + "  ".join(row) + " ⎦" + "\n"
        else:
            string_tensor += middle_bracket + "  ".join(row) + " ⎥" + "\n"
    return string_tensor


def assert_eq_tensors(computed: torch.Tensor, expected: torch.Tensor, term: str, header: Optional[list[str]] = None):
    header = [] if header is None else header
    result = (computed - expected).abs()
    assert (result <= 1e-5).all(), (
        f"{term} terms do not correspond to the expected ones (max(ε)={result.max()}).\n"
        f"{term}\n{header}\n{format_tensor(computed)}"
        f"\nExpected\n{format_tensor(expected)}\n"
        f"Diff\n{format_tensor(result)}"
    )


def test_vmax(methionine_model: MaudInput, methionine_allostery):
    model = Maudy(methionine_model)
    kcat_pars = model.maud_params.kcat.prior
    enzymes = [x.split("_")[0] for x in kcat_pars.ids[0]]
    kcat, enzyme_conc = methionine_allostery[-2:]
    enzyme_conc = enzyme_conc.loc[enzymes, :]
    assert (kcat.index == enzymes).all()
    vmax = get_vmax(torch.FloatTensor(kcat.to_numpy()), torch.FloatTensor(enzyme_conc.to_numpy()).T)
    return assert_eq_tensors(vmax, torch.FloatTensor(kcat.to_numpy() * enzyme_conc.to_numpy().T), "Vmax")


def test_FER_parity_with_methionine_model(methionine_model: MaudInput, methionine_allostery):
    model = Maudy(methionine_model)
    conc, km, ki, expected_free_enzyme_ratio = methionine_allostery[0:4]
    # check that the km/ki from Maud are aligned as expected in Maudy
    assert (km.index == model.maud_params.km.prior.ids[0]).all()
    assert (ki.index == model.maud_params.ki.prior.ids[0]).all()
    kcat_pars = model.maud_params.kcat.prior
    enzymatic_reactions = [x.split("_")[-1] for x in kcat_pars.ids[-1]]
    enzymes = [x.split("_")[0] for x in kcat_pars.ids[0]]
    edge_ids = [f"{e}_{r}" for e, r in zip(enzymes, enzymatic_reactions)]
    mics = [met.id for met in model.kinetic_model.mics]
    conc = torch.FloatTensor(conc.loc[mics, :].to_numpy().T)
    free_enzyme_ratio_denom =  get_free_enzyme_ratio_denom(
        conc,
        torch.FloatTensor(km),
        model.sub_conc_idx,
        model.sub_km_idx,
        model.prod_conc_idx,
        model.prod_km_idx,
        model.substrate_S,
        model.product_S,
        model.irreversible,
    )
    ci_denom = get_competitive_inhibition_denom(
        conc,
        torch.FloatTensor(ki),
        model.ki_conc_idx,
        model.ki_idx,
    )
    fer = 1 / (ci_denom + free_enzyme_ratio_denom)
    expected_free_enzyme_ratio = torch.Tensor(expected_free_enzyme_ratio.loc[:, edge_ids].to_numpy())
    assert_eq_tensors(fer, expected_free_enzyme_ratio, "FER", edge_ids)


def test_allostery_parity_with_methionine_maud_model(methionine_model: MaudInput, methionine_allostery):
    model = Maudy(methionine_model)
    conc, _, _, free_enzyme_ratio, tc, dc, expected_allostery, _, _ = methionine_allostery
    # we need to sort conc and FER as they are expected in Maudy
    kcat_pars = model.maud_params.kcat.prior
    mics = [met.id for met in model.kinetic_model.mics]
    conc = torch.FloatTensor(conc.loc[mics, :].to_numpy().T)
    enzymatic_reactions = [x.split("_")[-1] for x in kcat_pars.ids[-1]]
    enzymes = [x.split("_")[0] for x in kcat_pars.ids[0]]
    edge_ids = [f"{e}_{r}" for e, r in zip(enzymes, enzymatic_reactions)]
    # test if the model initialized FER and tc as expected
    allosteric_enzyme_names = [edge_ids[idx] for idx in model.allostery_reaction_idx]
    assert (
        free_enzyme_ratio.loc[:, edge_ids]
        .iloc[:, model.allostery_reaction_idx.numpy()]
        .columns.to_list()
        == allosteric_enzyme_names
    ), "FER must be aligned by model.allostery_reaction_idx"
    free_enzyme_ratio = torch.Tensor(free_enzyme_ratio.loc[:, edge_ids].to_numpy())
    assert tc.iloc[model.tc_idx.numpy()].index.to_list() == [
        enz.split("_")[0] for enz in allosteric_enzyme_names
    ], "Transfer constants must be aligned by model.allostery_reaction_idx"
    tc = torch.Tensor(tc.to_numpy())
    dc = torch.Tensor(dc.to_numpy())
    allostery = get_allostery(
        conc,
        free_enzyme_ratio,
        tc,
        dc,
        model.allostery_reaction_idx,
        model.d_to_reac_act,
        model.d_to_reac_inh,
        model.q_to_reac_act,
        model.q_to_reac_inh,
        model.conc_allostery_idx,
        model.tc_idx,
        model.subunits,
    )
    expected_allostery = torch.Tensor(expected_allostery.loc[edge_ids, :].to_numpy().T)
    assert_eq_tensors(allostery, expected_allostery, "Allostery", edge_ids)

