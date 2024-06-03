import torch
import torch.nn as nn


# Used in parameterizing p(flux | enz_conc)
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
        loc = torch.clamp(loc, -10, 10)  # Adjust the range as needed
        scale = torch.clamp(
            scale, 1e-6, 10
        )  # Ensure scale is positive and within a reasonable range

        return loc, scale
