"""Implement gradient analysis for metabolic control-like functionality."""

from functools import partial
from typing import Any

import torch
from torch.autograd.functional import jacobian

from .analysis import predict
from .model import Maudy
from .kinetics import compute_flux


DECODER_TO_SAMPLE = {
    "latent_bal_conc": "met",
    "unb_conc": "unb",
    "enzyme_conc": "enz",
    "kcat_drain": "drain",
}
PRIOR_VARS = {
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
}
EXP_INDEPENDENT = {"kcat", "dgf", "km", "psi", "ki", "dc", "tc", "fdx_contr", "dgr"}
# computed variables (not priors)
COMP_VARS = {"dgr", "ln_bal_conc"}


def get_jacobian(
    model: Maudy, prior_wrt: str = "enzyme_conc", d_conc: bool = True, samples: int = 1000, posterior: dict[str, torch.Tensor] | None = None,
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
    samples: int
        number of samples to draw from the posterior to calculate the jacobian on.
        Only used if `posterior` is None.
    posterior: dict[str, torch.Tensor] } | None, default=None
        samples generative from the posterior predictive distribution, which
        should encompass all of the sampling sites of the `Maudy.model`. If not
        specified, `model` will be sampled accordingly.

    Returns
    -------
    jacobian: torch.Tensor
        of shape [Experiment, N, prior_wrt] where N is num bal metabolites
        or num reactions; or, if `prior_wrt`
        is experiment-independent and there is only one experiment, [N, P]
    """
    assert (prior_wrt != "ln_bal_conc") or not d_conc, "dconc/dconc not possible, use control_matrices instead"
    # get sampled values from trained guide
    # guide_trace = poutine.trace(model.guide).get_trace(None, None, True, 1.0, True)
    var_names = tuple(set(DECODER_TO_SAMPLE.keys()) | PRIOR_VARS | COMP_VARS)
    if posterior is None:
        posterior = predict(model, samples, var_names)
    samples = next(iter(posterior.values())).shape[0]
    posterior = {site: t.squeeze(1) if site in EXP_INDEPENDENT else t for site, t in posterior.items()}
    # gather prior model variables
    prior_variable = posterior[prior_wrt]
    prior_val = prior_variable.detach().clone().requires_grad_(True)

    decoder_inputs = {
        nn_arg: posterior[site] if site in posterior else model.float_tensor([])
        for site, nn_arg in DECODER_TO_SAMPLE.items()
    }

    def forward(v, posterior, decoder_inputs):
        if prior_wrt == "ln_bal_conc":
            # neural networks are not involved
            nodes = posterior
            c_bal = v
        else:
            if prior_wrt in DECODER_TO_SAMPLE and d_conc:
                # the output is only dependant on the decoder
                c_bal = model.decoder(
                    **(decoder_inputs | {DECODER_TO_SAMPLE[prior_wrt]: v})
                )
            else:
                # the output dependends on both decoder and encoder
                nodes = posterior | {prior_wrt: v}
                encoder_inputs = _pack_encoder_inputs(nodes)
                x = model.concoder(**encoder_inputs)[-2]
                packed_inputs = decoder_inputs | {"met": x}
                c_bal = model.decoder(**packed_inputs)
            if d_conc:
                return c_bal
        conc = c_bal.new_ones(len(model.experiments), model.num_mics)
        conc[:, model.balanced_mics_idx] = model.safexp(c_bal)
        conc[:, model.unbalanced_mics_idx] = _get(nodes, "unb_conc")
        return compute_flux(
            model, conc, *[_get(nodes, site) for site in ["km", "ki", "kcat",
            "enzyme_conc", "dgr", "psi", "tc", "dc", "kcat_drain"]], 1e-9
        )  # fmt: skip

    def forward_i(posterior, decoder_inputs, i):
        posterior = {site: t[i] if t.numel() > 0 else t for site, t in posterior.items()}
        decoder_inputs = {site: t[i] if t.numel() > 0 else t for site, t in decoder_inputs.items()}
        return partial(forward, posterior=posterior, decoder_inputs=decoder_inputs)

    # this could be broadcasted but it's not since it would require to rearrange
    # the whole src/kinetics.py indexing, which is quite involved
    j: torch.Tensor = torch.stack([
        jacobian(forward_i(posterior, decoder_inputs, i), prior_val[i])  # type: ignore
        for i in range(prior_val.shape[0])
    ])
    pruned_jacobian = prune_jacobian(j)
    return pruned_jacobian


def _get(nodes: dict[str, Any], key: str) -> torch.Tensor:
    if key not in nodes:
        tensor = nodes["dgf"]
        return torch.tensor([], dtype=tensor.dtype, device=tensor.device)
    s = nodes[key]
    return s


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
    rest = [_get(nodes, site) for site in ["ki", "dc", "tc"] if site in nodes]
    encoder_inputs["rest"] = torch.cat(rest, dim=-1) if rest else _get(nodes, "rest")
    return encoder_inputs


def prune_jacobian(j: torch.Tensor) -> torch.Tensor:
    """Remove always-zero elements.

    These are the non-diagonal elements of experiment-dependent
    variables A with respect to the variables in B; i.e., the dependencies
    between two i.i.d. samples in different experiments (0).

    [N, E, A, E, B] -> [N, E, A, B]

    Parameters
    ----------
    j: torch.Tensor
        [N, E, A, E, B]
    """
    if j.ndim == 4:
        return j
    pruned = j.diagonal(dim1=1, dim2=3).movedim(-1, 1)
    return pruned


def control_matrices(model: Maudy, samples: int = 1000, posterior: dict[str, torch.Tensor] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
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
        $C^S$ [Samples, Experiments, Metabolites, Reactions] and $C^J$ [Samples, Experiments, Reactions, Reactions]
    """
    # we use the notation in Gunawardena 2002
    # balanced concentrations w.r.t. fluxes (both enzymatic and drains) (Eq. 28)
    elasticity = get_jacobian(model, "ln_bal_conc", False, samples=samples, posterior=posterior)
    N = model.S[model.balanced_mics_idx, :]
    return inverse_function(elasticity, N)


def inverse_function(elasticity: torch.Tensor, N: torch.Tensor):
    c_s = -torch.linalg.pinv(N @ elasticity, rcond=1e-9) @ N
    I = torch.eye(c_s.shape[-1], c_s.shape[-1]).unsqueeze(0)
    c_j = I + elasticity @ c_s
    return c_s, c_j


def mca(
    model: Maudy, prior_wrt: str = "enzyme_conc", d_conc: bool = True, samples: int = 1000
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
    torch.Tensor, [Samples, Experiments, A, B]:
        $\frac{\partial S}{\partial P}$ if `d_conc` else $\frac{\partial J}{\partial P}$.
    """
    var_names = tuple(set(DECODER_TO_SAMPLE.keys()) | PRIOR_VARS | COMP_VARS)
    posterior = predict(model, samples, var_names)
    c_s, c_j = control_matrices(model, posterior=posterior)
    v_wrt = get_jacobian(model, prior_wrt, False, posterior=posterior)
    c = c_s if d_conc else c_j
    return c @ v_wrt
