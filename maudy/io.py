import toml
from pathlib import Path
from typing import Optional

from .constants import T, DGF_WATER
from pydantic import BaseModel, Field


class NeuralNetworkConfig(BaseModel):
    met_dims: list[int] = [256, 256, 256, 256]
    km_dims: list[int] = [256, 256, 16]
    correction_dims: list[int] = [32, 64, 64, 32, 32]


class MaudyConfig(BaseModel):
    ferredoxin: Optional[dict[str, float]] = None
    neural_network: NeuralNetworkConfig = Field(default_factory=NeuralNetworkConfig)
    optimize_unbalanced_metabolites: list[str] = []
    correction_groups: list[list[str]] = []
    temperature: float = T
    dgf_water: float = DGF_WATER
    normalize: bool = False
    """whether to normalize the input and output of the neural network.
    The input is normalize such that positive values are turned to log-space
    and drains and fluxes are multiplied by 1e6 (mumol). The output is
    clamped between 0.6 orders of magnitude of the higher and lowest
    observed or prior concentrations.
    """


def load_maudy_config(maud_dir: Path) -> MaudyConfig:
    user_data = {}
    maudy_path = maud_dir / "maudy.toml"
    if maudy_path.exists():
        user_data = toml.load(maudy_path)
    return MaudyConfig(**user_data)
