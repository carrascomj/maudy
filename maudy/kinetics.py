"""Functions to replicate the modular rate law from Maud."""

import torch
from maud.data_model.kinetic_model import Reaction, ReactionMechanism
from .constants import DGF_WATER, RT

Vector = torch.Tensor
Matrix = torch.Tensor


def get_conc_idx(reac_st: list[str]) -> torch.LongTensor:
    raise NotImplementedError("gotta do it")


def get_km_idx(reac_st: list[str]) -> torch.LongTensor:
    raise NotImplementedError("gotta do it")


def get_enzyme_idx(reac: str) -> torch.LongTensor:
    raise NotImplementedError("gotta do it")


def get_drain_idx(reac: str) -> torch.LongTensor:
    raise NotImplementedError("gotta do it")


def get_dgr(S: Matrix, dgf: Vector, water_S: Vector) -> Vector:
    return S.T @ dgf + water_S * DGF_WATER


def get_saturation(
    conc: Vector,
    km: Vector,
    free_enzyme_ratio: Vector,
    reactions: list[Reaction],
) -> Vector:
    sub_conc_over_km = [
        (
            conc[
                get_conc_idx([met for met, st in reac.stoichiometry.items() if st < 0])
            ]
            / km[get_km_idx([met for met, st in reac.stoichiometry.items() if st < 0])]
        ).prod()
        if reac.mechanism != ReactionMechanism.drain
        else 1
        for reac in reactions
    ]
    return free_enzyme_ratio * sub_conc_over_km


def get_free_enzyme_ratio(
    conc: Vector,
    km: Vector,
    reactions: list[Reaction],
) -> Vector:
    prod_contr = (
        torch.Tensor([
            (
                1
                + (
                    conc[
                        get_conc_idx([
                            met for met, st in reac.stoichiometry.items() if st > 0
                        ])
                    ]
                    / km[
                        get_km_idx([
                            met for met, st in reac.stoichiometry.items() if st > 0
                        ])
                    ]
                )
                ** torch.FloatTensor([
                    st for st in reac.stoichiometry.values() if st > 0
                ])
            ).prod()
            if reac.mechanism != ReactionMechanism.reversible_michaelis_menten
            else 0
            for reac in reactions
        ])
        - 1
    )
    sub_contr = torch.Tensor([
        (
            1
            + (
                conc[
                    get_conc_idx([
                        met for met, st in reac.stoichiometry.items() if st < 0
                    ])
                ]
                / km[
                    get_km_idx([
                        met for met, st in reac.stoichiometry.items() if st < 0
                    ])
                ]
            )
            ** torch.FloatTensor([st for st in reac.stoichiometry.values() if st > 0])
        ).prod()
        if reac.mechanism != ReactionMechanism.drain
        else 1
        for reac in reactions
    ])
    return 1 / (sub_contr + prod_contr)


def get_reversibility(
    S: Matrix, dgr: Vector, conc: Vector
) -> Vector:
    return 1 - (dgr + RT * conc.log() @ S)


def get_vmax(
    kcat: Vector,
    enzyme_conc: Vector,  # , reactions: list[Reaction],
) -> Vector:
    return enzyme_conc * kcat
    # return kcat * enzyme_conc[[get_enzyme_idx(reac.id) for reac in reactions]]


def get_drains(drain_values: Vector, reactions: list[Reaction]) -> Vector:
    return torch.Tensor([
        drain_values[get_drain_idx(reac.id)]
        if reac.mechanism == ReactionMechanism.drain
        else 1
        for reac in reactions
    ])


def get_fluxes(
    conc: Vector,
    enzyme_conc: Vector,
    dgr: Vector,
    kcat: Vector,
    km: Vector,
    S: Matrix,
    drains: Vector,
    reactions: list[Reaction],
) -> Vector:
    free_enzyme_ratio = get_free_enzyme_ratio(conc, km, reactions)
    return (
        get_vmax(kcat, enzyme_conc)
        * get_saturation(conc, km, free_enzyme_ratio, reactions)
        * get_reversibility(S, dgr, conc, reactions)
        * get_drains(drains, reactions)
    )
