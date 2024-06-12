"""Neural networks used in the guide to approximate the ODE solver."""

import torch
import torch.nn as nn


class BaseConcCoder(nn.Module):
    """Base neural network, outputs location and scale of balanced metabolites."""

    def __init__(
        self,
        met_dims: list[int],
        reac_dims: list[int],
        km_dims: list[int],
    ):
        super().__init__()
        # metabolites
        self.met_dims = met_dims
        self.met_backbone = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Linear(in_dim, out_dim), nn.ReLU()
                )
                for in_dim, out_dim in zip(met_dims[:-1], met_dims[1:])
            ]
        )
        # reactions
        self.reac_backbone = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Linear(in_dim, out_dim), nn.ReLU()
                )
                for in_dim, out_dim in zip(reac_dims[:-1], reac_dims[1:])
            ]
        )
        # before passing it to the reac_backbone, we need to perform a convolution
        # over the reaction features (dgr, enz_conc, etc.) to a single feature
        self.reac_conv = nn.Sequential(nn.Conv1d(2, 1, 1), nn.ReLU())
        # kms
        self.km_backbone = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Linear(in_dim, out_dim), nn.ReLU()
                )
                for in_dim, out_dim in zip(km_dims[:-1], km_dims[1:])
            ]
        )

        self.emb_layer = nn.Sequential(
            nn.Linear(reac_dims[-1] + met_dims[-1] + km_dims[-1], met_dims[-2]),
            nn.ReLU(),
            nn.Linear(met_dims[-2], met_dims[-1]),
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
        conc: torch.FloatTensor,
        dgr: torch.FloatTensor,
        enz_conc: torch.FloatTensor,
        km: torch.FloatTensor,
    ) -> list[torch.Tensor]:
        out = self.met_backbone(conc)
        reac_features = torch.stack(
            [dgr.repeat(enz_conc.shape[0]).reshape(-1, dgr.shape[0]), enz_conc], dim=1
        )
        reac_features = self.reac_conv(reac_features)
        reac_features = reac_features.squeeze(1)
        reac_out = self.reac_backbone(reac_features)
        km_out = self.km_backbone(km.repeat(enz_conc.shape[0]).reshape(-1, km.shape[0]))
        out = self.emb_layer(torch.cat([out, reac_out, km_out], dim=-1))
        out = (out - torch.mean(out)) / torch.std(out)
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
