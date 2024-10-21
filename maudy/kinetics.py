"""Functions to replicate the modular rate law from Maud."""

from typing import Sequence

import torch
from .constants import DGF_WATER, F, RT

# per experiment expected type, just for clarification,
# although these functions are called with the experiments
# batched in the firs dimension
Vector = torch.Tensor
Matrix = torch.Tensor
# length reactions, each vector are indices that map a variable to that reaction
ReacIndex = Sequence[torch.LongTensor]


def get_dgr(
    S: Matrix, dgf: Vector, water_S: Vector, fdx_S: Vector, fdx_contr: torch.FloatTensor
) -> Vector:
    """Compute the experiment-agnostic dGr (no membrane potential)."""
    return S.T @ dgf + water_S * DGF_WATER + fdx_S * fdx_contr


def get_saturation(
    conc: Vector,
    km: Vector,
    free_enzyme_ratio: Vector,
    sub_conc_idx: list[Vector],
    sub_km_idx: list[Vector],
) -> Vector:
    sub_conc_over_km = torch.stack(
        [
            (conc[:, conc_idx] / km[..., km_idx]).prod(dim=-1)
            for conc_idx, km_idx in zip(sub_conc_idx, sub_km_idx)
        ],
        dim=1,
    )
    return free_enzyme_ratio * sub_conc_over_km


def get_free_enzyme_ratio_denom(
    conc: Vector,
    km: Vector,
    sub_conc_idx: list[Vector],
    sub_km_idx: list[Vector],
    prod_conc_idx: list[Vector],
    prod_km_idx: list[Vector],
    substrate_S: list[Vector],
    product_S: list[Vector],
    irr: Vector,
) -> Vector:
    prod_contr = torch.stack(
        [
            ((1.0 + (conc[:, conc_idx] / km[..., km_idx])) ** st).prod(dim=-1) - 1.0
            if km_idx.size(0)
            else torch.zeros(
                conc.shape[0], device=conc.device
            )  # if irr: no km for prods: no prod_contr
            for conc_idx, km_idx, st in zip(prod_conc_idx, prod_km_idx, product_S)
        ],
        dim=1,
    ) * irr  # 0 == irreversible, no information from products
    sub_contr = torch.stack(
        [
            ((1 + (conc[:, conc_idx] / km[..., km_idx])) ** st).prod(dim=-1)
            for conc_idx, km_idx, st in zip(sub_conc_idx, sub_km_idx, substrate_S)
        ],
        dim=1,
    )
    return sub_contr + prod_contr


def get_competitive_inhibition_denom(
    conc: Vector, ki: Vector, ci_conc_idx: ReacIndex, ki_idx: ReacIndex
):
    sub_contr = torch.stack(
        [
            (1 + (conc[:, conc_idx] / ki[..., ki_idx])).sum(dim=-1)
            for conc_idx, ki_idx in zip(ci_conc_idx, ki_idx)
        ],
        dim=1,
    )
    return sub_contr


def get_reversibility(
    S: Matrix, dgr: Vector, conc: Vector, trans_charge: Vector, psi: torch.Tensor,
    irr: Vector,
) -> Vector:
    """Add the membrane potential to dgr and compute reversibility."""
    rev = torch.ones((conc.shape[0], dgr.shape[0]), device=dgr.device)
    rev[:, irr] = 1 - torch.exp(
        (
            dgr[irr].unsqueeze(0)
            + trans_charge[irr].unsqueeze(0) * psi * F
            + RT * (conc.log() @ S)[:, irr]
        )
        / RT
    )
    return rev


def get_vmax(kcat: Vector, enzyme_conc: Vector) -> Vector:
    return enzyme_conc * kcat


def get_kinetic_multi_drain(
    kcat_drain: Vector,
    conc: Vector,
    sub_conc_idx: list[Vector],
    prod_conc_idx: list[Vector],
    substrate_S: list[Vector],
    product_S: list[Vector],
    drain_small_conc_corrector: float,
):
    """Multiply subs and prods to avoid having negative concentrations."""
    sub_contr = torch.stack(
        [
            (
                (conc[:, conc_idx] / (conc[:, conc_idx] + drain_small_conc_corrector))
                ** st
            ).prod(dim=-1)
            if conc_idx.size(0)
            else torch.ones(kcat_drain.shape[0], device=kcat_drain.device)
            for conc_idx, st in zip(sub_conc_idx, substrate_S)
        ],
        dim=1,
    )
    prod_contr = torch.stack(
        [
            (
                (conc[:, conc_idx] / (conc[:, conc_idx] + drain_small_conc_corrector))
                ** st
            ).prod(dim=-1)
            if conc_idx.size(0)
            else torch.ones(kcat_drain.shape[0], device=kcat_drain.device)
            for conc_idx, st in zip(prod_conc_idx, product_S)
        ],
        dim=1,
    )
    return kcat_drain * sub_contr * prod_contr


def get_allostery(
    conc: Vector,
    free_enzyme_ratio: Vector,
    transfer: Vector,  # only one per enzyme
    dissociation: Vector,  # only one per enzyme (either act or inh)
    reac_idx: torch.LongTensor,  # index of reactions to reaction with allosterism
    d_to_act: torch.LongTensor,  # index from act to reactions with allosterim (-1 if allosterism but not act)
    d_to_inh: torch.LongTensor,  # index from inh to reactions with allosterim (-1 if allosterism but not inh)
    conc_idx: ReacIndex,         # from concentrations to inh/act dissociations
    tc_idx: torch.LongTensor,    # index from transfer to reactions with allosterim
    subunits: Vector,
):
    out = torch.ones_like(free_enzyme_ratio)
    allostery = conc[:, conc_idx] / dissociation

    # calculate Q
    N = len(reac_idx)
    num = torch.ones((out.shape[0], N), dtype=allostery.dtype, device=allostery.device)
    denom = torch.ones((out.shape[0], N), dtype=allostery.dtype, device=allostery.device)
    # -1 in any index indicate no allostery modification
    num_mask = d_to_inh != -1
    denom_mask = d_to_act != -1
    num[:, d_to_inh[num_mask]] += allostery[:, d_to_inh[num_mask]]
    denom[:, d_to_act[denom_mask]] += allostery[:, d_to_act[denom_mask]]

    q = num / denom
    out[..., reac_idx] = 1 / (
        1
        + transfer[..., tc_idx]
        * (free_enzyme_ratio[..., reac_idx] * q) ** subunits
    )
    return out


def compute_flux(
    model,
    conc, km, ki,
    kcat, enz_conc,
    dgr, psi,
    tc, dc,
    kcat_drain, drain_km
):
    """Compute all the flux terms."""
    free_enz_km_denom = get_free_enzyme_ratio_denom(
        conc, km,
        model.sub_conc_idx, model.sub_km_idx,
        model.prod_conc_idx, model.prod_km_idx,
        model.substrate_S, model.product_S,
    )
    free_enz_ki_denom = get_competitive_inhibition_denom(
        conc, ki, model.ki_conc_idx, model.ki_idx
    ) if model.has_ci else 0
    free_enzyme_ratio = 1 / (free_enz_km_denom + free_enz_ki_denom)
    vmax = get_vmax(kcat, enz_conc)
    rev = get_reversibility(
        model.S_enz_thermo, dgr, conc, model.transported_charge, psi
    )
    sat = get_saturation(
        conc, km, free_enzyme_ratio, model.sub_conc_idx, model.sub_km_idx
    )
    allostery = (
        get_allostery(
            conc, free_enzyme_ratio,
            tc,
            dc,
            model.allostery_activation, model.allostery_idx,
            model.conc_allostery_idx, model.subunits,
        )
        if model.has_allostery
        else torch.ones_like(vmax)
    ).reshape(sat.shape[0], len(model.sub_km_idx))
    flux = vmax * rev * sat * allostery 
    drain = (
        get_kinetic_multi_drain(
            kcat_drain, conc,
            model.sub_conc_drain_idx, model.prod_conc_drain_idx,
            model.substrate_drain_S, model.product_drain_S,
            drain_km,
        )
        if kcat_drain.size()[0]
        else None
    )
    return torch.cat([drain, flux], dim=1) if drain is not None else flux
