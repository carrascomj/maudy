"""Neural networks used in the guide to approximate the ODE solver."""

from typing import Optional
import torch
import torch.nn as nn


class Clamp(nn.Module):
    def __init__(self, min_val: float, max_val: float):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.clamp(self.min_val, self.max_val)


class BaseConcCoder(nn.Module):
    """Base neural network, outputs location and scale of balanced metabolites."""

    def __init__(
        self,
        met_dims: list[int],
        reac_dim: int,
        km_dims: list[int],
        drain_dim: int = 0,
        ki_dim: int = 0,
        tc_dim: int = 0,
        obs_flux_dim: int = 0,
        drop_out: bool = False,
        batchnorm: bool = True,
        normalize: Optional[tuple[float, float]] = None,
    ):
        super().__init__()
        # metabolites
        self.met_dims = met_dims
        # reactions
        n_enz_reac = reac_dim
        reac_dim += drain_dim
        # kms
        constant_dims = km_dims.copy()
        constant_dims[0] = constant_dims[0] + 2 * n_enz_reac + ki_dim + tc_dim * 2
        self.constant_backbone = nn.Sequential(
            *[
                nn.Sequential(nn.Linear(in_dim, out_dim), nn.SiLU())
                for in_dim, out_dim in zip(constant_dims[:-1], constant_dims[1:])
            ],
        )
        emb_dims = met_dims.copy()
        emb_dims[0] = reac_dim + met_dims[0] + obs_flux_dim
        self.emb_layer = nn.Sequential(
            *[
                nn.Sequential(nn.Linear(in_dim, out_dim), nn.SiLU())
                for in_dim, out_dim in zip(emb_dims[:-1], emb_dims[1:])
            ],
            nn.Dropout1d() if drop_out else nn.Identity(),
        )

        out_dims = met_dims.copy()
        out_dims[0] = out_dims[-1]
        self.normalize = normalize is not None
        self.out_layers = nn.ModuleList(
            [
                nn.Sequential(  # loc layer
                    *[
                        nn.Sequential(
                            nn.Linear(in_dim, out_dim),
                            nn.BatchNorm1d(out_dim) if batchnorm else nn.Identity(),
                            nn.ReLU(),
                            nn.Dropout1d() if drop_out else nn.Identity(),
                        )
                        for in_dim, out_dim in zip(out_dims[:-1], out_dims[1:])
                    ],
                    nn.Linear(out_dims[-1], out_dims[-1]),
                    Clamp(normalize[0], normalize[1])
                    if normalize is not None
                    else nn.Identity(),
                ),
                nn.Sequential(  # scale layer
                    *[
                        nn.Sequential(
                            nn.Linear(in_dim, out_dim),
                            nn.BatchNorm1d(out_dim) if batchnorm else nn.Identity(),
                            nn.ReLU(),
                            nn.Dropout1d() if drop_out else nn.Identity(),
                        )
                        for in_dim, out_dim in zip(out_dims[:-1], out_dims[1:])
                    ],
                    nn.Linear(out_dims[-1], out_dims[-1]),
                    nn.Softplus(),  # makes the output positive
                ),
            ],
        )
        self._initialize_weights(self)

    def _initialize_weights(self, module):
        for m in module.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def set_dropout(self: nn.Module, p: float = 0.0):
        for m in self.modules():
            if isinstance(m, nn.Dropout1d):
                m.p = p

    def forward(
        self,
        conc: torch.Tensor,
        dgr: torch.Tensor,
        enz_conc: torch.Tensor,
        kcat: torch.Tensor,
        drains: torch.Tensor,
        km: torch.Tensor,
        rest: torch.Tensor,
        obs_flux: Optional[torch.Tensor] = None,
    ) -> list[torch.Tensor]:
        if self.normalize:
            enz_conc, drains, kcat, km, rest = (
                enz_conc.log(),
                drains * 1e6,
                kcat.log(),
                km.log(),
                rest.log(),
            )
        # these are all small numbers so we need to normalize
        # to avoid collapsing the batch dimension
        reac_in = torch.cat((enz_conc, drains), dim=1)
        constant_in = torch.cat([dgr, kcat, km, rest.flatten()])
        constant_q = self.constant_backbone(constant_in)
        if obs_flux is not None:
            k = self.emb_layer(torch.cat((conc, reac_in, obs_flux * 1e6), dim=1))
        else:
            k = self.emb_layer(torch.cat((conc, reac_in), dim=1))
        out = k * constant_q.unsqueeze(0)
        return [out_layer(out) for out_layer in self.out_layers]


def fdx_head(concoder: BaseConcCoder):
    """Add an (B, 1) output for Fdx contribution."""
    met_dims = concoder.met_dims
    fdx_layer = nn.Sequential(
        nn.Linear(met_dims[-1], met_dims[-2]),
        nn.Sigmoid(),
        nn.Linear(met_dims[-2], 1),
    )
    concoder.out_layers.append(fdx_layer)


def unb_opt_head(concoder: BaseConcCoder, unb_dim: int):
    """Add an (B, UnbOpt) output for optimized unbalanced metabolites."""
    met_dims = concoder.met_dims
    unb_met_loc_layer = nn.Sequential(
        nn.Linear(met_dims[-1], met_dims[-2]),
        nn.SiLU(),
        nn.Linear(met_dims[-2], unb_dim),
    )
    concoder.out_layers.append(unb_met_loc_layer)


class BaseDecoder(nn.Module):
    def __init__(
        self,
        met_dim: int,
        unb_dim: int,
        enz_dim: int,
        drain_dim: int,
        normalize: Optional[tuple[int, int]] = None,
    ):
        super().__init__()
        layer = nn.Linear(met_dim + unb_dim + enz_dim + drain_dim, met_dim)
        self.normalize = normalize is not None
        self.loc_layer = (
            nn.Sequential(layer, Clamp(normalize[0], normalize[1]))
            if normalize is not None
            else layer
        )

    def forward(
        self,
        met: torch.Tensor,
        unb: torch.Tensor,
        enz: torch.Tensor,
        drain: torch.Tensor,
    ):
        if self.normalize:
            met, unb, enz, drain = met.log(), unb.log(), enz.log(), drain * 1e6
        features = torch.cat((met, unb, enz, drain), dim=1)
        return self.loc_layer(features)
