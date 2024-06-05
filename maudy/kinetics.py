"""Functions to replicate the modular rate law from Maud."""

import torch
from .constants import DGF_WATER, F, RT

Vector = torch.Tensor
Matrix = torch.Tensor


def get_dgr(S: Matrix, dgf: Vector, water_S: Vector) -> Vector:
    """Compute the experiment-agnostic dGr (no membrane potential)."""
    return S.T @ dgf + water_S * DGF_WATER


def get_saturation(
    conc: Vector,
    km: Vector,
    free_enzyme_ratio: Vector,
    sub_conc_idx: list[Vector],
    sub_km_idx: list[Vector],
) -> Vector:
    sub_conc_over_km = torch.cat([
        (conc[:, conc_idx] / km[..., km_idx]).prod(dim=-1)
        for conc_idx, km_idx in zip(sub_conc_idx, sub_km_idx)
    ])
    return free_enzyme_ratio * sub_conc_over_km.reshape(conc.shape[0], -1)


def get_free_enzyme_ratio(
    conc: Vector,
    km: Vector,
    sub_conc_idx: list[Vector],
    sub_km_idx: list[Vector],
    prod_conc_idx: list[Vector],
    prod_km_idx: list[Vector],
    substrate_S: list[Vector],
    product_S: list[Vector],
) -> Vector:
    prod_contr = torch.cat([
        ((1.0 + (conc[:, conc_idx] / km[..., km_idx])) ** st).prod(dim=-1) - 1.0
        if km_idx.size(0)
        else torch.zeros(1)  # if irr: no km for prods: no prod_contr
        for conc_idx, km_idx, st in zip(prod_conc_idx, prod_km_idx, product_S)
    ])
    sub_contr = torch.cat([
        ((1 + (conc[:, conc_idx] / km[..., km_idx])) ** st).prod(dim=-1)
        for conc_idx, km_idx, st in zip(sub_conc_idx, sub_km_idx, substrate_S)
    ])
    return (1 / (sub_contr + prod_contr)).reshape(conc.shape[0], -1)


def get_reversibility(S: Matrix, dgr: Vector, conc: Vector, trans_charge: Vector, psi: torch.Tensor) -> Vector:
    """Add the membrane potential to dgr and compute reversibility."""
    return 1 - torch.exp((dgr.unsqueeze(0) + trans_charge.unsqueeze(0) * psi.unsqueeze(1) * F + RT * (conc.log() @ S)) / RT)


def get_vmax(kcat: Vector, enzyme_conc: Vector) -> Vector:
    return enzyme_conc * kcat
