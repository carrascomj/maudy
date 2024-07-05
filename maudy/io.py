import toml
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class NeuralNetworkConfig(BaseModel):
    met_dims: list[int] = [256, 256, 256, 256]
    km_dims: list[int] = [256, 256, 16]


class MaudyConfig(BaseModel):
    ferredoxin: Optional[dict[str, float]] = None
    neural_network: NeuralNetworkConfig = Field(default_factory=NeuralNetworkConfig)
    optimize_unbalanced_metabolites: list[str] = []


def load_maudy_config(maud_dir: Path) -> MaudyConfig:
    user_data = {}
    maudy_path = maud_dir / "maudy.toml"
    if maudy_path.exists():
        user_data = toml.load(maudy_path)
    return MaudyConfig(**user_data)
