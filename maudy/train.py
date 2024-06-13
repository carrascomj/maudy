"""Training loop and CLI."""

import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyro
import torch
from pyro.infer import SVI, TraceEnum_ELBO, config_enumerate
from pyro.optim import ClippedAdam
from tqdm import tqdm
from .model import Maudy
from maud.loading_maud_inputs import load_maud_input
from maud.data_model.maud_input import MaudInput


def log_cosine_schedule(start_value: float, end_value: float, iterations: int, min_value: float = float("-inf")) -> torch.Tensor:
    """Compute a log cosine schedule between `start_value` and `end_value` over `iterations`."""
    assert iterations > 0, ValueError("Number of iterations must be greater than 0.")  
    log_start = np.log10(start_value)
    log_end = np.log10(end_value)
    t = np.arange(iterations) / (iterations - 1)
    cos_values = (1 + np.cos(np.pi * t)) / 2
    log_schedule = log_end + (log_start - log_end) * cos_values
    # Convert back from log10 scale
    schedule = torch.FloatTensor(10 ** log_schedule)
    schedule[schedule < min_value] = min_value
    return schedule


def train(maud_input: MaudInput, num_epochs: int, penalize_ss: bool, eval_flux: bool, eval_conc: bool):
    pyro.clear_param_store()
    # Enable optional validation warnings
    pyro.enable_validation(False)

    # Instantiate instance of model/guide and various neural networks
    maudy = Maudy(maud_input, optimize_unbalanced=["pi_c", "atp_c", "adp_c"])
    # maudy.print_inputs()
    penalization_temp = log_cosine_schedule(0.001, 0.1, num_epochs, 0.001)
    if torch.cuda.is_available():
        maudy.cuda()
        penalization_temp = penalization_temp.cuda()
    obs_flux, obs_conc = maudy.get_obs()
    if not eval_flux:
        obs_flux = None
    if not eval_conc:
        obs_conc = None

    # Setup an optimizer (Adam) and learning rate scheduler.
    # We start with a moderately high learning rate (0.006) and
    # reduce to 6e-7 over the course of training.
    # optimizer = ClippedAdam({"lr": 0.0006, "lrd": 0.0001 ** (1 / num_epochs)})
    optimizer = ClippedAdam({"lr": 0.0006, "lrd": 0.0001 ** (1 / num_epochs)})
    # Tell Pyro to enumerate out y when y is unobserved.
    # (By default y would be sampled from the guide)
    guide = config_enumerate(maudy.guide, "parallel", expand=True)

    # Setup a variational objective for gradient-based learning.
    # Note we use TraceEnum_ELBO in order to leverage Pyro's machinery
    # for automatic enumeration of the discrete latent variable y.
    elbo = TraceEnum_ELBO(strict_enumeration_warning=False)
    svi = SVI(maudy.model, guide, optimizer, elbo)

    progress_bar = tqdm(penalization_temp, total=num_epochs, desc="Training", unit="epoch")
    for penalization_ss in progress_bar:
        loss = svi.step(obs_flux, obs_conc, penalization_ss if penalize_ss else False)
        lr = list(optimizer.get_state().values())[0]["param_groups"][0]["lr"]
        progress_bar.set_postfix(loss=f"{loss:.2e}", lr=f"{lr:.2e}", ss=f"{penalization_ss:.2e}")
    return maudy, optimizer


def get_timestamp():
    return datetime.now().isoformat().replace(":", "").replace("-", "").replace(".", "")


def load_ferredoxin(maud_dir: Path) -> Optional[dict[str, float]]:
    ferre_path = maud_dir / "ferredoxin.txt"
    if ferre_path.exists():
        return (
            pd.read_csv(ferre_path, sep=",", names=["reaction", "stoichiometry"])
            .set_index("reaction")
            .to_dict()["stoichiometry"]
        )


def sample(
    maud_dir: Path,
    num_epochs: int = 100,
    out_dir: Optional[Path] = None,
    penalize_ss: bool = True,
    eval_flux: bool = True,
    eval_conc: bool = True,
    smoke: bool = False,
):
    """Sample model."""
    maud_input = load_maud_input(str(maud_dir))
    maud_input._fdx_stoichiometry = load_ferredoxin(maud_dir)
    maudy, optimizer = train(maud_input, num_epochs, penalize_ss, eval_flux, eval_conc)
    if smoke:
        return
    out = (
        out_dir
        if out_dir is not None
        else Path(f"maudyout_{maud_input.config.name}_{get_timestamp()}")
    )
    os.mkdir(out)
    torch.save(
        {"maudy": maudy.state_dict(), "optimizer": optimizer.get_state()},
        out / "model.pt",
    )
    shutil.copytree(maud_dir, out / "user_input")
