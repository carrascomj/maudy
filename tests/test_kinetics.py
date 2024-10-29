"""Check kinetic parity with Maud."""

from maud.loading_maud_inputs import MaudInput
import torch
from maudy.model import Maudy
from maudy.kinetics import get_allostery


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


def test_allostery_parity_with_methionine_maud_model(methionine_model: MaudInput, methionine_allostery):
    model = Maudy(methionine_model)
    conc, free_enzyme_ratio, tc, dc, expected_allostery = methionine_allostery
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
    result = (expected_allostery - allostery).abs()
    assert (result <= 1e-5).all(), (
        f"Allosteric terms do not correspond to the expected ones (max(ε)={result.max()}).\n"
        f"Allostery\n{edge_ids}\n{format_tensor(allostery)}"
        f"\nExpected\n{format_tensor(expected_allostery)}\n"
        f"Diff\n{format_tensor(result)}"
    )

