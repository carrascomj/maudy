import torch
import torch.nn as nn


# Used in parameterizing p(bal_conc | unb_conc)
class ToyDecoder(nn.Module):
    def __init__(self, dims: list[int]):
        super().__init__()
        self.fc = nn.Sequential(
            *[
                nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU())
                for in_dim, out_dim in zip(dims[:-1], dims[1:])
            ]
        )
        self.out_layer = nn.Linear(dims[-1], dims[-1])
        self.fc_loc = nn.Sequential(
            nn.Linear(dims[-1], dims[-2]),
            nn.Sigmoid(),
            nn.Linear(dims[-2], dims[-1]),
            nn.Softplus(),
        )

    def forward(self, enz_conc) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.fc(enz_conc)
        loc = self.out_layer(out)
        scale = self.fc_loc(out)

        # Stabilization: clip the outputs to prevent extreme values
        loc = torch.clamp(loc, -25, 25)
        scale = torch.clamp(
            scale, 0.1, 3.0
        )  # Ensure scale is positive and within a reasonable range

        return loc, scale
