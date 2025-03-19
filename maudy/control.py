"""Implement gradient analysis for metabolic control-like functionality."""

from typing import Any

import torch
import pyro.poutine as poutine
from torch.autograd.functional import jacobian

from .model import Maudy
from .kinetics import compute_flux


DECODER_TO_SAMPLE = {
    "latent_bal_conc": "met",
    "unb_conc": "unb",
    "enzyme_conc": "enz",
    "kcat_drain": "drain",
}
PRIOR_VARS = [
    "kcat",
    "dgf",
    "km",
    "psi",
    "enzyme_conc",
    "kcat_drain",
    "unb_conc",
    "ki",
    "dc",
    "tc",
    "fdx_contr",
    "fdx_ratio",
]


def get_jacobian(
    model: Maudy, prior_wrt: str = "enzyme_conc", d_conc: bool = True
) -> torch.Tensor:
    r"""Generate gradients of the steady-state concentrations or fluxes w.r.t. a _prior variable_ `prior_wrt`.

    Prior variables refers to the prior model (kinetic paramters, enzyme conc, etc.).
    These prior variables are first sampled to then fixed them to generate a
    conditioned model. The jacobian is extracted exclusively from passing
    the output of the conditioned model to the decoder neural network.

    Parameters
    ----------
    model: Maudy
    prior_wrt: str
        names of sample site $p$ to compute the jacobian
        $\fraction{\partial [C]_b}{\partial p}$
    d_conc: bool, default=False
        if d_conc, the numerator are steady-state concentrations, fluxes otherwise.

    Returns
    -------
    jacobian: torch.Tensor
        of shape [Experiment, N, prior_wrt] where N is num bal metabolites
        or num reactions; or, if `prior_wrt`
        is experiment-independent and there is only one experiment, [N, P]
    """
    assert (prior_wrt != "ln_bal_conc") or not d_conc, "dconc/dconc not possible, use control_matrices instead"
    # get sampled values from trained guide
    guide_trace = poutine.trace(model.guide).get_trace(None, None, True, 1.0, True)
    # gather prior model variables
    # the model might not have this sampling sites in case there are no drains, no CI, etc.
    prior_vars = [prior for prior in PRIOR_VARS if prior in guide_trace.nodes]

    # fix latent variables to the traced output
    conditioned_model = poutine.condition(
        model.model, data={k: guide_trace.nodes[k]["value"] for k in prior_vars}
    )

    # sample from conditioned model
    model_trace = poutine.trace(conditioned_model).get_trace(
        None, None, True, 1.0, True
    )

    # finally, calculate jacobian of the steady concentrations w.r.t. prior_wrt
    prior_variable = model_trace.nodes[prior_wrt]["value"]
    prior_val = prior_variable.detach().clone().requires_grad_(True)

    decoder_inputs = {
        nn_arg: model_trace.nodes[site]["value"]
        if site in model_trace.nodes
        else model.float_tensor([])
        for site, nn_arg in DECODER_TO_SAMPLE.items()
    }

    def forward(v):
        if prior_wrt == "ln_bal_conc":
            # neural networks are not involved
            nodes = model_trace.nodes
            c_bal = v
        else:
            if prior_wrt in DECODER_TO_SAMPLE and d_conc:
                # the output is only dependant on the decoder
                c_bal = model.decoder(
                    **(decoder_inputs | {DECODER_TO_SAMPLE[prior_wrt]: v})
                )
            else:
                # the output dependends on both decoder and encoder
                nodes = model_trace.nodes | {prior_wrt: v}
                encoder_inputs = _pack_encoder_inputs(nodes)
                x = model.concoder(**encoder_inputs)[-2]
                c_bal = model.decoder(**(decoder_inputs | {"met": x}))
            if d_conc:
                return c_bal
        conc = c_bal.new_ones(len(model.experiments), model.num_mics)
        conc[:, model.balanced_mics_idx] = model.safexp(c_bal)
        conc[:, model.unbalanced_mics_idx] = _get(nodes, "unb_conc")
        return compute_flux(
            model, conc, *[_get(nodes, site) for site in ["km", "ki", "kcat",
            "enzyme_conc", "dgr", "psi", "tc", "dc", "kcat_drain"]], 1e-9
        )  # fmt: skip

    j: torch.Tensor = jacobian(forward, prior_val)
    pruned_jacobian = prune_jacobian(j)
    return pruned_jacobian


def _get(nodes: dict[str, Any], key: str) -> torch.Tensor:
    if key not in nodes:
        tensor = nodes["dgf"]["value"]
        return torch.tensor([], dtype=tensor.dtype, device=tensor.device)
    s = nodes[key]
    return s if isinstance(s, torch.Tensor) else s["value"]


def _pack_encoder_inputs(nodes: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Pack inputs given nodes.

    This is required since the inputs of the concoder are not 1:1 map from
    the sample sites.

    Parameters
    ----------
    nodes: dict[str, Any]
        the values are either from a trace (message with key `value`) or
        directly a torch.Tensor.
    """
    encoder_inputs = {nn_arg: _get(nodes, nn_arg) for nn_arg in ["dgr", "kcat", "km"]}
    # and the special cases
    encoder_inputs["conc"] = torch.log(_get(nodes, "unb_conc"))
    encoder_inputs["enz_conc"] = _get(nodes, "enzyme_conc")
    encoder_inputs["drains"] = _get(nodes, "kcat_drain")
    rest = _get(nodes, "rest")  # empty tensor
    encoder_inputs["rest"] = torch.cat(
        [rest] + [_get(nodes, site) for site in ["ki", "dc", "tc"] if site in nodes]
    )
    return encoder_inputs


def prune_jacobian(j: torch.Tensor) -> torch.Tensor:
    if len(j.shape) == 3:
        # if experiment-independent, the j.shape is [1, N, C]
        j.squeeze_(0)
        return j
    # prune jacobian of unrelated rows, extracting the diagonal
    # [E, A, E, B] -> [E, A, B]
    pruned_jacobian = j[torch.arange(j.shape[0]), :, torch.arange(j.shape[0]), :]
    return pruned_jacobian


def control_matrices(model: Maudy) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Get concentration and flux control matrix.

    Following ["Notes on Metabolic Control Analysis" by Gunawardena 2002](http://jeremy-gunawardena.com/papers/mca.pdf),
    we get Eq. 25:

    $$
    C^S = - (N \frac{\partial v}{\partial S})^{-1} N,
    $$

    where $C^S$ is the control matrix, $N$ is the stoichoimatric matrix, $v$ is the flux
    vector and $s$ is the concentration vector. We also return the flux control matrix (Eq. 26):

    $$
    C^J = I - \frac{\partial v}{\partial S} (N \frac{\partial v}{\partial S})^{-1} N.
    $$

    Returns
    -------
    (c_s, c_j): tuple[torch.Tensor, torch.Tensor]
        $C^S$ [Experiments, Metabolites, Reactions] and $C^J$ [Experiments, Reactions, Reactions]
    """
    # we use the notation in Gunawardena 2002
    # balanced concentrations w.r.t. fluxes (both enzymatic and drains) (Eq. 28)
    elasticity = get_jacobian(model, "ln_bal_conc", False)
    N = model.S.T[:, model.balanced_mics_idx].permute(1, 0)
    c_s = -torch.inverse(N @ elasticity) @ N
    I = torch.eye(c_s.shape[-1], c_s.shape[-1]).unsqueeze(0)
    c_j = I + elasticity @ c_s
    return c_s, c_j


def mca(
    model: Maudy, prior_wrt: str = "enzyme_conc", d_conc: bool = True
) -> torch.Tensor:
    r"""Get metabolic control analysis for `prior_wrt`.

    Following ["Notes on Metabolic Control Analysis" by Gunawardena 2002](http://jeremy-gunawardena.com/papers/mca.pdf),
    we get Eq. 27:

    $$
    \frac{\partial S}{\partial P} = C^S \frac{\partial v}{\partial P}
    $$

    or

    $$
    \frac{\partial J}{\partial P} = C^J \frac{\partial v}{\partial P}
    $$

    where $C^S$ is the concentration control matrix and $C^J$ is the flux control matrix (see `control_matrices`).

    Returns
    -------
    torch.Tensor:
        $\frac{\partial S}{\partial P}$ if `d_conc` else $\frac{\partial J}{\partial P}$.
    """
    c_s, c_j = control_matrices(model)
    v_wrt = get_jacobian(model, prior_wrt, False)
    c = c_s if d_conc else c_j
    return c @ v_wrt
