"""Training loop and CLI."""

import os
import shutil
from typer import Option
from datetime import datetime
from pathlib import Path
from typing import Annotated, Optional

import pyro
import torch
from pyro.infer import SVI, TraceEnum_ELBO, config_enumerate
from pyro.optim.clipped_adam import ClippedAdam
from pyro.optim import PyroOptim
from tqdm import tqdm
from maud.loading_maud_inputs import load_maud_input
from maud.data_model.maud_input import MaudInput

from .io import load_maudy_config
from .model import Maudy


def anneal(epoch, annealing_epochs, min_factor):
    """Get linear interpolation between 1 and `min_factor` at `epoch`."""
    return (
        min_factor
        + (1.0 - min_factor) * (float(epoch + 1) / float(annealing_epochs))
        if annealing_epochs > 0 and epoch < annealing_epochs
        else 1.0
    )


def train(
    maud_input: MaudInput,
    num_epochs: int,
    penalize_ss: bool,
    eval_flux: bool,
    eval_conc: bool,
    annealing_epochs: int,
    normalize: bool,
):
    pyro.clear_param_store()
    # Enable optional validation warnings
    pyro.enable_validation(False)

    # Instantiate instance of model/guide and various neural networks
    maudy = Maudy(maud_input, normalize)
    if torch.cuda.is_available():
        maudy.cuda()
    obs_flux, obs_conc = maudy.get_obs()
    if not eval_flux:
        obs_flux = None
    if not eval_conc:
        obs_conc = None

    lr_start = 3e-4
    lr_end = 8e-5
    optimizer = PyroOptim(
        ClippedAdam,
        optim_args={"lrd": (lr_end / lr_start) ** (1 / num_epochs), "lr": lr_start},
    )
    # Tell Pyro to enumerate out y when y is unobserved.
    # (By default y would be sampled from the guide)
    guide = config_enumerate(maudy.guide, "parallel", expand=True)

    # Setup a variational objective for gradient-based learning.
    # Note we use TraceEnum_ELBO in order to leverage Pyro's machinery
    # for automatic enumeration of the discrete latent variable y.
    elbo = TraceEnum_ELBO(strict_enumeration_warning=False)
    svi = SVI(maudy.model, guide, optimizer, elbo)

    progress_bar = tqdm(range(num_epochs), desc="Training", unit="epoch")
    for epoch in progress_bar:
        t = anneal(epoch, annealing_epochs, 0.2)
        loss = svi.step(obs_flux, obs_conc, penalize_ss, t)
        lr = optimizer.get_state()['km_scale']["param_groups"][0]["lr"]
        progress_bar.set_postfix(loss=f"{loss:+.2e}", lr=f"{lr:.2e}", T=f"{t:.2f}")
    return maudy, optimizer


def get_timestamp():
    return datetime.now().isoformat().replace(":", "").replace("-", "").replace(".", "")


def sample(
    maud_dir: Path,
    num_epochs: int = 100,
    annealing_stage: Annotated[float, Option(help="Part of training that will be annealing the KL")] = 0.2,
    normalize: Annotated[bool, Option(help="Whether to normalize input and output of NN")] = False,
    out_dir: Optional[Path] = None,
    penalize_ss: bool = True,
    eval_flux: bool = True,
    eval_conc: bool = True,
    smoke: bool = False,
):
    """Sample model."""
    maud_input = load_maud_input(str(maud_dir))
    maud_input._maudy_config = load_maudy_config(maud_dir)
    maudy, optimizer = train(maud_input, num_epochs, penalize_ss, eval_flux, eval_conc, int(num_epochs * annealing_stage), normalize)
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
