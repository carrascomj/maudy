"""Neural networks used in the guide to approximate the ODE solver."""

from typing import Optional

import jax.numpy as jnp
from jax import jit
from flax import linen as nn


@jit
def norm(x):
    return (x - jnp.expand_dims(jnp.mean(x, axis=1), axis=1)) / jnp.expand_dims(jnp.sqrt((jnp.var(x, axis=1) + 1e-11)), axis=1)


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
    batchnorm: bool = True  # not actually a batchnorm, just a normalization over batch
    normalize: Optional[tuple[float, float]] = None
    train: bool = True
    has_fdx: bool = False

    @nn.compact
    def __call__(self, conc, dgr, enz_conc, kcat, drains, km, rest):
        if self.normalize:
            conc, enz_conc, drains, kcat, km, rest = (
                jnp.log(conc),
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
        constant_q = nn.silu(nn.Dense(self.km_dims[0])(constant_in))
        constant_q = nn.silu(nn.Dense(self.km_dims[1])(constant_q))
        constant_q = nn.Dense(self.km_dims[-1])(constant_q)
        exp_in = jnp.concat((conc, reac_in), axis=1)
        k = nn.silu(nn.Dense(self.met_dims[1])(exp_in))
        k = nn.silu(nn.Dense(self.met_dims[2])(k))
        k = nn.silu(nn.Dense(self.met_dims[3])(k))
        k = nn.Dense(features=self.met_dims[-1])(k)
        out = k * jnp.expand_dims(constant_q, axis=0)
        loc = out
        scale = out
        for out_dim in self.met_dims[1:]:
            if self.batchnorm:
                loc = nn.Sequential([nn.Dense(out_dim), norm, nn.relu, nn.Dropout(0.5, deterministic=not self.train)])(loc)
                scale = nn.Sequential([nn.Dense(out_dim), norm, nn.relu, nn.Dropout(0.5, deterministic=not self.train)])(scale)
            else:
                loc = nn.Sequential([nn.Dense(out_dim), nn.relu, nn.Dropout(0.5, deterministic=not self.train)])(loc)
                scale = nn.Sequential([nn.Dense(out_dim), nn.relu, nn.Dropout(0.5, deterministic=not self.train)])(scale)
        if self.normalize is not None:
            loc = loc.clip(self.normalize[0], self.normalize[1])
        scale = nn.softplus(scale)
        fdx = nn.Sequential([nn.Dense(self.met_dims[-2]), nn.sigmoid, nn.Dense(1)])(out) if self.has_fdx else jnp.zeros((3,1))
        return loc, scale, fdx


class BaseDecoder(nn.Module):
    met_dim: int
    unb_dim: int
    enz_dim: int
    drain_dim: int
    normalize: Optional[tuple[int, int]] = None

    @nn.compact
    def __call__(self, met, unb, enz, drain):
        if self.normalize:
            unb, enz, drain = jnp.log(unb), jnp.log(enz), drain * 1e6
        features = (met, unb, enz, drain) if drain.size != 0 else (met, unb, enz)
        features = jnp.concat(features, axis=1)
        loc = nn.Dense(self.met_dim)(features)
        if self.normalize is not None:
            loc = jnp.clip(loc, self.normalize[0], self.normalize[1])
        return loc
