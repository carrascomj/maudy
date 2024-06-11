from typing import Optional
import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(
        self,
        met_dims: list[int],
        reac_dims: list[int],
        drain_dims: list[int],
        S: torch.Tensor,
        reac_features: int,
    ):
        super().__init__()
        self.met_backbone = nn.Sequential(
            *[
                nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU())
                for in_dim, out_dim in zip(met_dims[:-1], met_dims[1:])
            ]
        )
        self.reac_backbone = nn.Sequential(
            *[
                nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU())
                for in_dim, out_dim in zip(reac_dims[:-1], reac_dims[1:])
            ]
        )
        # before passing it to the reac_backbone, we need to perform a convolution
        # over the reaction features (dgr, enz_conc, etc.) to a single feature
        self.reac_conv = nn.Sequential(nn.Conv1d(reac_features, 1, 1), nn.ReLU())

        self.drain_backbone = (
            nn.Sequential(
                *[
                    nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU())
                    for in_dim, out_dim in zip(drain_dims[:-1], drain_dims[1:])
                ]
            )
            if drain_dims[0]
            else None
        )
        self.S = S
        self.loc_layer = nn.Linear(met_dims[-1], met_dims[-1])
        self.scale_layer = nn.Sequential(
            nn.Linear(met_dims[-1], met_dims[-2]),
            nn.Sigmoid(),
            nn.Linear(met_dims[-2], met_dims[-1]),
            nn.Softplus(),
        )

    def forward(
        self,
        unb_conc: torch.FloatTensor,
        dgr: torch.FloatTensor,  # shape [B, R]
        enz_conc: torch.FloatTensor,  # shape [B, R]
        balanced_mics_idx: torch.LongTensor,
        drains: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.met_backbone(unb_conc)
        # convolution over all reac features (dgr, enz_conc)
        reac_features = torch.stack(
            [dgr.repeat(enz_conc.shape[0]).reshape(-1, dgr.shape[0]), enz_conc], dim=1
        )
        reac_features = self.reac_conv(reac_features)
        reac_features = reac_features.squeeze(1)

        reac_out = self.reac_backbone(reac_features)
        if drains is not None:
            assert self.drain_backbone is not None
            drain_out = self.drain_backbone(drains)
            reac_out = torch.cat([drain_out, reac_out], dim=-1)
        reac_out = (reac_out @ self.S.T)[..., balanced_mics_idx]
        out = out + reac_out

        loc = self.loc_layer(out)
        scale = self.scale_layer(out)

        return loc, scale


class Encoder(nn.Module):
    def __init__(
        self,
        reac_dims: list[int],
        met_dims: list[int],
        S: torch.Tensor,
    ):
        super().__init__()
        self.reac_backbone = nn.Sequential(
            *[
                nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU())
                for in_dim, out_dim in zip(reac_dims[:-1], reac_dims[1:])
            ]
        )
        self.S = S
        self.met_out = nn.Sequential(
            *[
                nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU())
                for in_dim, out_dim in zip(met_dims[:-1], met_dims[1:])
            ]
        )
        self.loc_layer = nn.Linear(met_dims[-1], met_dims[-1])
        self.scale_layer = nn.Sequential(
            nn.Linear(met_dims[-1], met_dims[-2]),
            nn.Sigmoid(),
            nn.Linear(met_dims[-2], met_dims[-1]),
            nn.Softplus(),
        )

    def forward(
        self,
        flux: torch.FloatTensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.reac_backbone(flux) @ self.S.T
        out = self.met_out(out)
        loc = self.loc_layer(out)

        return loc


class ConcCoder(nn.Module):
    def __init__(
        self,
        met_dims: list[int],
        **_,
    ):
        super().__init__()
        self.met_backbone = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU()
                )
                for in_dim, out_dim in zip(met_dims[:-1], met_dims[1:])
            ]
        )
        self.loc_layer = nn.Sequential(nn.Linear(met_dims[-1], met_dims[-1]))
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        conc: torch.FloatTensor,
        *_args,
        **_kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.met_backbone(conc)
        loc = self.loc_layer(out)
        scale = self.scale_layer(out)
        return loc, scale


class ConcFdxCoder(nn.Module):
    def __init__(
        self,
        met_dims: list[int],
    ):
        super().__init__()
        self.met_backbone = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU()
                )
                for in_dim, out_dim in zip(met_dims[:-1], met_dims[1:])
            ]
        )
        self.loc_layer  = nn.Sequential(
            nn.Linear(met_dims[-1], met_dims[-2]),
            nn.Sigmoid(),
            nn.Linear(met_dims[-2], met_dims[-1]),
        )
        self.scale_layer = nn.Sequential(nn.Linear(met_dims[-1], met_dims[-1]), nn.Softplus())
        self.fdx_layer = nn.Sequential(
            nn.Linear(met_dims[-1], met_dims[-2]),
            nn.Sigmoid(),
            nn.Linear(met_dims[-2], 1),
            nn.Softplus(),
        )
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        conc: torch.FloatTensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.met_backbone(conc)
        loc = self.loc_layer(out)
        scale = self.scale_layer(out)
        fdx_contribution = self.fdx_layer(out)
        return loc, scale, fdx_contribution


class AllFdxCoder(nn.Module):
    def __init__(
        self,
        met_dims: list[int],
        reac_dims: list[int],
        km_dims: list[int],
        **_,
    ):
        super().__init__()
        # metabolites
        self.met_backbone = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU()
                )
                for in_dim, out_dim in zip(met_dims[:-1], met_dims[1:])
            ]
        )
        # reactions
        self.reac_backbone = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU()
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
                    nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU()
                )
                for in_dim, out_dim in zip(km_dims[:-1], km_dims[1:])
            ]
        )

        self.out_layer = nn.Sequential(
            nn.Linear(reac_dims[-1] + met_dims[-1] + km_dims[-1], met_dims[-2]),
            nn.ReLU(),
            nn.Linear(met_dims[-2], met_dims[-1]),
        )
        self.loc_layer  = nn.Sequential(
            nn.Linear(met_dims[-1], met_dims[-2]),
            nn.ReLU(),
            nn.Linear(met_dims[-2], met_dims[-1]),
        )
        self.scale_layer = nn.Sequential(nn.Linear(met_dims[-1], met_dims[-1]), nn.Softplus())
        self.fdx_layer = nn.Sequential(
            nn.Linear(met_dims[-1], met_dims[-2]),
            nn.Sigmoid(),
            nn.Linear(met_dims[-2], 1),
        )

    def forward(
        self,
        conc: torch.FloatTensor,
        dgr: torch.FloatTensor,
        enz_conc: torch.FloatTensor,
        km: torch.FloatTensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.met_backbone(conc)
        reac_features = torch.stack(
            [dgr.repeat(enz_conc.shape[0]).reshape(-1, dgr.shape[0]), enz_conc], dim=1
        )
        reac_features = self.reac_conv(reac_features)
        reac_features = reac_features.squeeze(1)
        reac_out = self.reac_backbone(reac_features)
        km_out = self.km_backbone(km.repeat(enz_conc.shape[0]).reshape(-1, km.shape[0]))
        out = self.out_layer(torch.cat([out, reac_out, km_out], dim=-1))

        loc = self.loc_layer(out)
        scale = self.scale_layer(out)
        fdx_contribution = self.fdx_layer(out)
        return loc, scale, fdx_contribution


class AllFdxUnbCoder(AllFdxCoder):
    def __init__(
        self,
        met_dims: list[int],
        reac_dims: list[int],
        km_dims: list[int],
        unb_dims: int,
    ):
        super().__init__(met_dims, reac_dims, km_dims)
        # metabolites
        self.unb_met_loc_layer   = nn.Sequential(
            nn.Linear(met_dims[-1], met_dims[-2]),
            nn.ReLU(),
            nn.Linear(met_dims[-2], unb_dims),
        )

    def forward(
        self,
        conc: torch.FloatTensor,
        dgr: torch.FloatTensor,
        enz_conc: torch.FloatTensor,
        km: torch.FloatTensor
    ) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        out = self.met_backbone(conc)
        reac_features = torch.stack(
            [dgr.repeat(enz_conc.shape[0]).reshape(-1, dgr.shape[0]), enz_conc], dim=1
        )
        reac_features = self.reac_conv(reac_features)
        reac_features = reac_features.squeeze(1)
        reac_out = self.reac_backbone(reac_features)
        km_out = self.km_backbone(km.repeat(enz_conc.shape[0]).reshape(-1, km.shape[0]))
        out = self.out_layer(torch.cat([out, reac_out, km_out], dim=-1))

        loc = self.loc_layer(out)
        scale = self.scale_layer(out)
        fdx_contribution = self.fdx_layer(out)
        loc_unb = self.unb_met_loc_layer(out)
        return (loc, scale, fdx_contribution), loc_unb
