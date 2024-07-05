"""Neural networks used in the guide to approximate the ODE solver."""

from typing import Optional

import jax.numpy as jnp
from flax import linen as nn


class BaseConcCoder(nn.Module):
    """Base neural network, outputs location and scale of balanced metabolites."""

    met_dims: list[int]
    reac_dim: int
    km_dims: list[int]
    drain_dim: int = 0
    ki_dim: int = 0
    tc_dim: int = 0
    obs_flux_dim: int = 0
    drop_out: bool = False
    batchnorm: bool = True
    normalize: Optional[tuple[float, float]] = None
    train: bool = True

    @nn.compact
    def __call__(self, conc, dgr, enz_conc, kcat, drains, km, rest):
        if self.normalize:
            enz_conc, drains, kcat, km, rest = (
                jnp.log(enz_conc),
                drains * 1e6,
                jnp.log(kcat),
                jnp.log(km),
                jnp.log(rest),
            )
        # these are all small numbers so we need to normalize
        # to avoid collapsing the batch dimension
        reac_in = (
            jnp.concat((enz_conc, drains), axis=1) if drains.size != 0 else enz_conc
        )
        constant_in = jnp.concat([dgr, kcat, km, rest.flatten()])
        constant_q = nn.silu(nn.Dense(features=self.km_dims[0])(constant_in))
        constant_q = nn.silu(nn.Dense(features=self.km_dims[1])(constant_q))
        constant_q = nn.Dense(features=self.km_dims[-1])(constant_q)
        exp_in = jnp.concat((conc, reac_in), axis=1)
        k = nn.relu(nn.Dense(features=self.met_dims[1])(exp_in))
        k = nn.relu(nn.Dense(features=self.met_dims[2])(k))
        k = nn.relu(nn.Dense(features=self.met_dims[3])(k))
        k = nn.Dense(features=self.met_dims[-1])(k)
        out = k * jnp.expand_dims(constant_q, axis=0)
        loc = out
        scale = out
        for out_dim in self.met_dims[1:]:
            loc = nn.Sequential([nn.Dense(out_dim), nn.relu, nn.Dropout(0.5, deterministic=not self.train)])(loc)
            scale = nn.Sequential([nn.Dense(out_dim), nn.relu, nn.Dropout(0.5, deterministic=not self.train)])(scale)
        if self.normalize is not None:
            loc = loc.clip(self.normalize[0], self.normalize[1])
        scale = nn.softplus(scale)
        return loc, scale


# def fdx_head(concoder: BaseConcCoder):
#     """Add an (B, 1) output for Fdx contribution."""
#     met_dims = concoder.met_dims
#     fdx_layer = nn.Sequential(
#         nn.Dense(met_dims[-2]),
#         nn.sigmoid,
#         nn.Dense(1),
#     )
#     concoder.out_layers.append(fdx_layer)


# def unb_opt_head(concoder: BaseConcCoder, unb_dim: int):
#     """Add an (B, UnbOpt) output for optimized unbalanced metabolites."""
#     met_dims = concoder.met_dims
#     unb_met_loc_layer = nn.Sequential(
#         nn.Dense(met_dims[-2]),
#         nn.silu,
#         nn.Dense(unb_dim),
#     )
#     concoder.out_layers.append(unb_met_loc_layer)


class BaseDecoder(nn.Module):
    met_dim: int
    unb_dim: int
    enz_dim: int
    drain_dim: int
    normalize: Optional[tuple[int, int]] = None

    @nn.compact
    def __call__(self, met, unb, enz, drain):
        if self.normalize:
            met, unb, enz, drain = jnp.log(met), jnp.log(unb), jnp.log(enz), drain * 1e6
        features = (met, unb, enz, drain) if drain.size != 0 else (met, unb, enz)
        features = jnp.concat(features, axis=1)
        loc = nn.Dense(self.met_dim)(features)
        if self.normalize is not None:
            loc = jnp.clip(loc, self.normalize[0], self.normalize[1])
        return loc
