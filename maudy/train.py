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
from maud.loading_maud_inputs import load_maud_input
from maud.data_model.maud_input import MaudInput

from .io import load_maudy_config
from .model import Maudy


def train(
    maud_input: MaudInput,
    num_epochs: int,
    penalize_ss: bool,
    eval_flux: bool,
    eval_conc: bool,
):
    pyro.clear_param_store()
    # Enable optional validation warnings
    pyro.enable_validation(False)

    # Instantiate instance of model/guide and various neural networks
    maudy = Maudy(maud_input)
    if torch.cuda.is_available():
        maudy.cuda()
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

    progress_bar = tqdm(range(num_epochs), desc="Training", unit="epoch")
    for _ in progress_bar:
        loss = svi.step(obs_flux, obs_conc, penalize_ss)
        lr = list(optimizer.get_state().values())[0]["param_groups"][0]["lr"]
        progress_bar.set_postfix(loss=f"{loss:.2e}", lr=f"{lr:.2e}")
    return maudy, optimizer


def get_timestamp():
    return datetime.now().isoformat().replace(":", "").replace("-", "").replace(".", "")


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
    maud_input._maudy_config = load_maudy_config(maud_dir)
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
    pyro.get_param_store().save(str(out / "model_params.pt"))
    shutil.copytree(maud_dir, out / "user_input")
