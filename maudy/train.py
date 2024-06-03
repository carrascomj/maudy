"""Training loop and CLI."""

import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

import pyro
import torch
from pyro.infer import SVI, TraceEnum_ELBO, config_enumerate
from pyro.optim import ClippedAdam
from typer import run
from .model import Maudy
from maud.loading_maud_inputs import load_maud_input
from maud.data_model.maud_input import MaudInput


def train(maud_input: MaudInput, num_epochs: int, gpu: bool = False):
    pyro.clear_param_store()
    # Enable optional validation warnings
    pyro.enable_validation(True)

    # Instantiate instance of model/guide and various neural networks
    maudy = Maudy(maud_input)
    maudy.print_inputs()
    if gpu:
        maudy = maudy.cuda()

    obs_fluxes, obs_conc = maudy.get_obs()

    # Setup an optimizer (Adam) and learning rate scheduler.
    # We start with a moderately high learning rate (0.006) and
    # reduce by a factor of 5 after 20 epochs.
    optimizer = ClippedAdam({"lr": 0.006, "lrd": 0.2 ** (1 / num_epochs)})

    # Tell Pyro to enumerate out y when y is unobserved.
    # (By default y would be sampled from the guide)
    guide = config_enumerate(maudy.guide, "parallel", expand=True)

    # Setup a variational objective for gradient-based learning.
    # Note we use TraceEnum_ELBO in order to leverage Pyro's machinery
    # for automatic enumeration of the discrete latent variable y.
    elbo = TraceEnum_ELBO(strict_enumeration_warning=False)
    svi = SVI(maudy.model, guide, optimizer, elbo)

    for epoch in range(num_epochs):
        loss = svi.step(obs_fluxes, obs_conc)
        print(f"[Epoch {epoch}]  Loss: {loss:.2e}")

    return maudy, optimizer


def get_timestamp():
    return datetime.now().isoformat().replace(":", "").replace("-", "").replace(".", "")


def sample(
    maud_dir: Path,
    num_epochs: int = 100,
    out_dir: Optional[Path] = None,
    smoke: bool = False,
):
    """Sample model."""
    maud_input = load_maud_input(str(maud_dir))
    maudy, optimizer = train(maud_input, num_epochs)
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
