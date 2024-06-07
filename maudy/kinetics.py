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
) -> Vector:
    prod_contr = torch.stack(
        [
            ((1.0 + (conc[:, conc_idx] / km[..., km_idx])) ** st).prod(dim=-1) - 1.0
            if km_idx.size(0)
            else torch.zeros(1)  # if irr: no km for prods: no prod_contr
            for conc_idx, km_idx, st in zip(prod_conc_idx, prod_km_idx, product_S)
        ],
        dim=1,
    )
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
    S: Matrix, dgr: Vector, conc: Vector, trans_charge: Vector, psi: torch.Tensor
) -> Vector:
    """Add the membrane potential to dgr and compute reversibility."""
    return 1 - torch.exp(
        (
            dgr.unsqueeze(0)
            + trans_charge.unsqueeze(0) * psi.unsqueeze(1) * F
            + RT * (conc.log() @ S)
        )
        / RT
    )


def get_vmax(kcat: Vector, enzyme_conc: Vector) -> Vector:
    return enzyme_conc * kcat


def get_kinetic_drain(
    kcat_drain: Vector,
    conc: Vector,
    sub_conc_idx: list[Vector],
    drain_small_conc_corrector: float,
):
    sub_contr = torch.stack(
        [
            (conc[:, conc_idx] / (conc[:, conc_idx] + drain_small_conc_corrector)).prod(
                dim=-1
            )
            for conc_idx in sub_conc_idx
        ],
        dim=1,
    )
    return kcat_drain * sub_contr
