"""Training loop and CLI."""

import pyro
from pyro.infer import SVI, TraceEnum_ELBO, config_enumerate
from pyro.optim import ClippedAdam
from pathlib import Path
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

    # Setup an optimizer (Adam) and learning rate scheduler.
    # We start with a moderately high learning rate (0.006) and
    # reduce by a factor of 5 after 20 epochs.
    scheduler = ClippedAdam({"lr": 0.006, "lrd": 0.2 ** (1 / num_epochs)})


    # Tell Pyro to enumerate out y when y is unobserved.
    # (By default y would be sampled from the guide)
    guide = config_enumerate(maudy.guide, "parallel", expand=True)

    # Setup a variational objective for gradient-based learning.
    # Note we use TraceEnum_ELBO in order to leverage Pyro's machinery
    # for automatic enumeration of the discrete latent variable y.
    elbo = TraceEnum_ELBO(strict_enumeration_warning=False)
    svi = SVI(maudy.model, guide, scheduler, elbo)

    for epoch in range(num_epochs):
        loss = svi.step()
        # scheduler.step()
        print(f"[Epoch {epoch}]  Loss: {loss:.2e}")


def app(maud_dir: Path, num_epochs: int = 100):
    """Train model."""
    maud_input = load_maud_input(str(maud_dir))
    train(maud_input, num_epochs)


def main():
    run(app)
