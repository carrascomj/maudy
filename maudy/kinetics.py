"""Functions to replicate the modular rate law from Maud."""

from typing import Sequence

import jax
import jax.numpy as jnp
from .constants import DGF_WATER, F, RT

# per experiment expected type, just for clarification,
# although these functions are called with the experiments
# batched in the firs dimension
Vector = jnp.ndarray
Matrix = jnp.ndarray
# length reactions, each vector are indices that map a variable to that reaction
ReacIndex = Sequence[jnp.ndarray]


def get_dgr(
    S: Matrix, dgf: Vector, water_S: Vector, fdx_S: Vector, fdx_contr: jnp.ndarray
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
    if conc.ndim == 1:
        conc = conc[None, :]
    if km.ndim == 1:
        km = km[None, :]
    sub_conc_over_km = jnp.stack(
        [
            (conc[:, conc_idx] / km[..., km_idx]).prod(axis=-1)
            for conc_idx, km_idx in zip(sub_conc_idx, sub_km_idx)
        ],
        axis=1,
    )
    sub_conc_over_km = sub_conc_over_km if sub_conc_over_km.shape == free_enzyme_ratio.shape else sub_conc_over_km.squeeze(-1)
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
    # Ensure conc and km have at least 2 dimensions
    if conc.ndim == 1:
        conc = conc[None, :]
    if km.ndim == 1:
        km = km[None, :]

    # Get the batch size
    batch_size = conc.shape[0]

    def process_prod_contr(conc_idx, km_idx, st):
        if km_idx.size != 0:
            result = ((1.0 + (conc[:, conc_idx] / km[:, km_idx])) ** st).prod(axis=-1) - 1.0
        else:
            result = jnp.zeros(batch_size)
        # Ensure the result has the shape (batch_size, 1)
        return result.reshape(batch_size, -1)

    def process_sub_contr(conc_idx, km_idx, st):
        result = ((1 + (conc[:, conc_idx] / km[:, km_idx])) ** st).prod(axis=-1)
        # Ensure the result has the shape (batch_size, 1)
        return result.reshape(batch_size, -1)

    prod_contr = jnp.concatenate(
        [
            process_prod_contr(conc_idx, km_idx, st)
            for conc_idx, km_idx, st in zip(prod_conc_idx, prod_km_idx, product_S)
        ],
        axis=1,
    )

    sub_contr = jnp.concatenate(
        [
            process_sub_contr(conc_idx, km_idx, st)
            for conc_idx, km_idx, st in zip(sub_conc_idx, sub_km_idx, substrate_S)
        ],
        axis=1,
    )

    return sub_contr + prod_contr

def get_competitive_inhibition_denom(
    conc: Vector, ki: Vector, ci_conc_idx: ReacIndex, ki_idx: ReacIndex
) -> Vector:
    # Ensure conc and ki have at least 2 dimensions
    if conc.ndim == 1:
        conc = conc[None, :]
    if ki.ndim == 1:
        ki = ki[None, :]

    # Get the batch size
    batch_size = conc.shape[0]

    def process_contr(conc_idx, ki_idx):
        if conc_idx.size != 0 and ki_idx.size != 0:
            result = (1 + (conc[:, conc_idx] / ki[:, ki_idx])).sum(axis=-1)
        else:
            result = jnp.zeros(batch_size)
        # Ensure the result has the shape (batch_size, 1)
        return result.reshape(batch_size, 1)

    sub_contr = jnp.concatenate(
        [
            process_contr(conc_idx, ki_idx)
            for conc_idx, ki_idx in zip(ci_conc_idx, ki_idx)
        ],
        axis=1,
    )

    return sub_contr


def get_reversibility(
    S: Matrix, dgr: Vector, conc: Vector, trans_charge: Vector, psi: jnp.ndarray
) -> Vector:
    """Add the membrane potential to dgr and compute reversibility."""
    return 1 - jnp.exp(
        (
            jnp.expand_dims(dgr, axis=0)
            + jnp.expand_dims(trans_charge, axis=0) * psi * F
            + RT * (jnp.log(conc) @ S)
        )
        / RT
    )


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
    sub_contr = jnp.stack(
        [
            (
                (conc[:, conc_idx] / (conc[:, conc_idx] + drain_small_conc_corrector))
                ** st
            ).prod(axis=-1)
            if conc_idx.size == 0
            else jnp.ones(kcat_drain.shape[0])
            for conc_idx, st in zip(sub_conc_idx, substrate_S)
        ],
        axis=1,
    )
    prod_contr = jnp.stack(
        [
            (
                (conc[:, conc_idx] / (conc[:, conc_idx] + drain_small_conc_corrector))
                ** st
            ).prod(axis=-1)
            if conc_idx.size == 0
            else jnp.ones(kcat_drain.shape[0])
            for conc_idx, st in zip(prod_conc_idx, product_S)
        ],
        axis=1,
    )
    return kcat_drain * sub_contr * prod_contr


def get_allostery(
    conc: Vector,
    free_enzyme_ratio: Vector,
    transfer: Vector,  # only one per enzyme
    dissociation: Vector,  # only one per enzyme (either act or inh)
    is_activation: jnp.ndarray,  # 1 if act; 0 if inh
    reac_idx: jnp.ndarray,  # index of corresponding reactions
    conc_idx: ReacIndex,
    subunits: Vector,
):
    if conc.ndim == 1:
        conc = conc[None, :]
    out = jnp.ones_like(free_enzyme_ratio)
    allostery = conc[:, conc_idx] / dissociation
    act_denom = 1.0 + is_activation * allostery
    inh_num = 1.0 + ~is_activation * allostery
    out.at[..., reac_idx].set(1 / (
        1
        + transfer
        * (free_enzyme_ratio[..., reac_idx] * inh_num / act_denom) ** subunits
    ))
    return out


def compute_flux(
    conc, enz_conc, kcat_drain, km, ki, kcat, dgr, psi, tc, dc, drain_km,
    sub_conc_idx, sub_km_idx, prod_conc_idx, prod_km_idx,
    substrate_S, product_S, allostery_activation,
    allostery_idx, conc_allostery_idx, subunits,
    ki_conc_idx, ki_idx, S_enz_thermo, transported_charge,
    has_ci, has_allostery, product_drain_S
):
    """Compute all the flux terms."""
    free_enz_km_denom = get_free_enzyme_ratio_denom(
        conc,
        km,
        sub_conc_idx,
        sub_km_idx,
        prod_conc_idx,
        prod_km_idx,
        substrate_S,
        product_S,
    )
    free_enz_ki_denom = (
        get_competitive_inhibition_denom(conc, ki, ki_conc_idx, ki_idx)
        if has_ci
        else 0
    )
    free_enzyme_ratio = 1 / (free_enz_km_denom + free_enz_ki_denom)
    vmax = get_vmax(kcat, enz_conc)
    rev = get_reversibility(
        S_enz_thermo, dgr, conc, transported_charge, psi
    )
    sat = get_saturation(
        conc, km, free_enzyme_ratio, sub_conc_idx, sub_km_idx
    )
    allostery = (
        get_allostery(
            conc,
            free_enzyme_ratio,
            tc,
            dc,
            allostery_activation,
            allostery_idx,
            conc_allostery_idx,
            subunits,
        )
        if has_allostery
        else jnp.ones_like(vmax)
    ).reshape(sat.shape[0], len(sub_km_idx))
    flux = vmax * rev * sat * allostery
    drain = (
        get_kinetic_multi_drain(
            kcat_drain,
            conc,
            sub_conc_drain_idx,
            prod_conc_drain_idx,
            substrate_drain_S,
            product_drain_S,
            drain_km,
        )
        if kcat_drain.size != 0
        else None
    )
    return jnp.concat([drain, flux], axis=1) if drain is not None else flux


def dc_dt(
    conc, _t, enz_conc, drain, theta_shared
):
    """Differentiate the concentration of balanced metabolites."""
    theta_exp = enz_conc, drain
    theta_shared, S, balanced_mics_idx, unbalanced_mics_idx = theta_shared[:-3], theta_shared[-3], theta_shared[-2], theta_shared[-1]
    flux = compute_flux(conc, *theta_exp, *theta_shared)
    dc = (flux @ S.T).at[unbalanced_mics_idx].set(0.0)
    return dc


def pack_theta(model, km, ki, kcat, enz_conc, dgr, psi, tc, dc, kcat_drain, drain_km):
    theta_shared = (km, ki, kcat, dgr, psi, tc, dc, drain_km,
        model.sub_conc_idx, model.sub_km_idx, model.prod_conc_idx, model.prod_km_idx, model.substrate_S,
        model.product_S, model.allostery_activation, model.allostery_idx, model.conc_allostery_idx,
        model.subunits, model.ki_conc_idx, model.ki_idx, model.S_enz_thermo,
        model.transported_charge, model.has_ci, model.has_allostery,
        model.product_drain_S, model.S, model.balanced_mics_idx, model.unbalanced_mics_idx
    )
    # variables different for each exp
    theta_exp = enz_conc if kcat_drain.size == 0 else (enz_conc, kcat_drain)
    return theta_exp, theta_shared


