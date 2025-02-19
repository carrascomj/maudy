"""Implement gradient analysis for metabolic control-like functionality."""

import torch

import pyro.poutine as poutine
from torch.autograd.functional import jacobian

from .model import Maudy


DECODER_TO_SAMPLE = {
    "latent_bal_conc": "met",
    "unb_conc": "unb",
    "enzyme_conc": "enz",
    "kcat_drain": "drain",
}


def get_jacobian(model: Maudy, prior_wrt: str = "enzyme_conc") -> torch.Tensor:
    r"""Generate gradients of the concentration w.r.t. a _prior variable_ `prior_wrt`.

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
    
    Returns
    -------
    jacobian: torch.Tensor
        of shape [Experiment, Balanced Metabolite, prior_wrt]
    """
    # get sampled values from trained guide
    guide_trace = poutine.trace(model.guide).get_trace(None, None, True, 1.0, True)
    # gather prior model variables
    prior_vars = [
        "kcat",
        "dgf",
        "km",
        "psi",
        "enzyme_conc",
        "kcat_drain",
        "unb_conc",
        "ci",
        "dc",
        "tc",
        "fdx_contr",
        "fdx_ratio",
    ]
    # the model might not have this sampling sites in case there are no drains, no CI, etc.
    prior_vars = [prior for prior in prior_vars if prior in guide_trace.nodes]

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

    nn_inputs = {
        nn_arg: model_trace.nodes[site]["value"]
        if site in model_trace.nodes
        else model.float_tensor([])
        for site, nn_arg in DECODER_TO_SAMPLE.items()
    }

    def forward(v):
        return model.decoder(**(nn_inputs | {DECODER_TO_SAMPLE[prior_wrt]: v}))

    j: torch.Tensor = jacobian(forward, prior_val)
    # prune jacobian of full-zero rows, corresponding to prior_wrt gradient
    #  w.r.t. concentrations in other experiments
    pruned_jacobian = j[~torch.all(j == 0, dim=-1)]
    # recover the experiment dim that was flattened
    return pruned_jacobian.reshape(j.shape[0], -1, pruned_jacobian.shape[-1])
