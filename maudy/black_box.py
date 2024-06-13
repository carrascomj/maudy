"""Neural networks used in the guide to approximate the ODE solver."""

from typing import Optional
import torch
import torch.nn as nn


class BaseConcCoder(nn.Module):
    """Base neural network, outputs location and scale of balanced metabolites."""

    def __init__(
        self,
        met_dims: list[int],
        reac_dims: list[int],
        km_dims: list[int],
        drain_dim: int = 0,
        ki_dim: int = 0,
        tc_dim: int = 0,
    ):
        super().__init__()
        # metabolites
        self.met_dims = met_dims
        self.met_backbone = nn.Sequential(*[
            nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU())
            for in_dim, out_dim in zip(met_dims[:-1], met_dims[1:])
        ])
        # reactions
        reac_dims[0] += drain_dim
        self.reac_backbone = nn.Sequential(*[
            nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU())
            for in_dim, out_dim in zip(reac_dims[:-1], reac_dims[1:])
        ])
        # before passing it to the reac_backbone, we need to perform a convolution
        # over the reaction features (dgr, enz_conc, etc.) to a single feature
        self.reac_conv = nn.Sequential(nn.Conv1d(3, 1, 1), nn.ReLU())
        # kms
        self.km_backbone = nn.Sequential(*[
            nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU())
            for in_dim, out_dim in zip(km_dims[:-1], km_dims[1:])
        ])

        self.emb_layer = nn.Sequential(
            nn.Linear(
                reac_dims[-1] + met_dims[-1] + km_dims[-1] + ki_dim + tc_dim * 2,
                met_dims[-2],
            ),
            nn.ReLU(),
            nn.Linear(met_dims[-2], met_dims[-1]),
            nn.SiLU(),
        )
        self.out_layers = nn.ModuleList(
            [
                nn.Sequential(  # loc layer
                    nn.Linear(met_dims[-1], met_dims[-2]),
                    nn.ReLU(),
                    nn.Linear(met_dims[-2], met_dims[-1]),
                ),
                # scale layer
                nn.Sequential(nn.Linear(met_dims[-1], met_dims[-1]), nn.Softplus()),
            ],
        )

    def forward(
        self,
        conc: torch.Tensor,
        dgr: torch.Tensor,
        enz_conc: torch.Tensor,
        kcat: torch.Tensor,
        drains: torch.Tensor,
        km: torch.Tensor,
        rest: torch.Tensor,
    ) -> list[torch.Tensor]:
        out = self.met_backbone(conc)
        enz_reac_features = torch.stack(
            [dgr.repeat(enz_conc.shape[0]).reshape(-1, dgr.shape[0]), kcat.repeat(enz_conc.shape[0]).reshape(-1, kcat.shape[0]), enz_conc], dim=1
        )
        enz_reac_features = self.reac_conv(enz_reac_features)
        enz_reac_features = enz_reac_features.squeeze(1)
        reac_out = self.reac_backbone(torch.cat([enz_reac_features, drains]))
        km_out = self.km_backbone(km.repeat(enz_conc.shape[0]).reshape(-1, km.shape[0]))
        out = self.emb_layer(
            torch.cat(
                [
                    out,
                    reac_out,
                    km_out,
                    rest.repeat(enz_conc.shape[0]).reshape(enz_conc.shape[0], rest.shape[0]),
                ],
                dim=-1,
            )
        )
        out = (out - torch.mean(out, dim=-1)) / torch.std(out)
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
        nn.ReLU(),
        nn.Linear(met_dims[-2], unb_dim),
    )
    concoder.out_layers.append(unb_met_loc_layer)
