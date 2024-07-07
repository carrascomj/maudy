"""Training loop and CLI."""

import dill as pickle
import os
import shutil
from typer import Option
from datetime import datetime
from pathlib import Path
from typing import Annotated, Optional

from jax import jit, random, numpy as jnp
from tqdm import tqdm
from maud.loading_maud_inputs import load_maud_input
from maud.data_model.maud_input import MaudInput
import optax

import numpyro
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.svi import SVIState

from .io import load_maudy_config
from .model import Maudy
from .analysis import predict, print_summary_dfs, report_to_dfs


# Custom loss function
class CustomTrace_ELBO(Trace_ELBO):
    def loss(self, rng_key, param_map, model, guide, *args, **kwargs):
        t = args[-1]  # Extract annealing factor from arguments
        elbo_loss = super().loss(rng_key, param_map, model, guide, *args, **kwargs)
        guide_trace = numpyro.handlers.trace(guide).get_trace(*args, **kwargs)
        model_trace = numpyro.handlers.trace(numpyro.handlers.replay(model, guide_trace)).get_trace(*args, **kwargs)
        
        init_conc = guide_trace['latent_bal_conc']['value']
        steady_state_conc = model_trace['bal_conc']['value']
        
        # Compute the additional loss term
        # additional_loss = jnp.mean((init_conc - steady_state_conc)**2)
        
        return elbo_loss #+ additional_loss * t


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
) -> tuple[Maudy, dict]:
    """Train the model.
    
    Returns
    -------
    tuple[Maudy, dict]
        the model object (contains model and guide) and the optimized params dictionary
    """

    # Instantiate instance of model/guide and various neural networks
    maudy = Maudy(maud_input, normalize)
    obs_flux, obs_conc = maudy.get_obs()
    if not eval_flux:
        obs_flux = None
    if not eval_conc:
        obs_conc = None

    lr_start = 3e-4
    lr_end = 8e-5
    learning_rate_schedule = optax.linear_schedule(init_value=lr_start, end_value=lr_end, transition_steps=num_epochs)
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),  # Clip gradients by global norm
        optax.adam(learning_rate_schedule)
    )
    elbo = Trace_ELBO()
    svi = SVI(maudy.model, maudy.guide, optimizer, elbo)
    state = svi.init(rng_key=random.PRNGKey(23), obs_flux=obs_flux, obs_conc=obs_conc, penalize_ss=penalize_ss, annealing_factor=0.1)
    state = SVIState(
        optim_state=state.optim_state,
        mutable_state=state.mutable_state,
        rng_key=state.rng_key,
    )

    @jit
    def update_state(state, obs_flux, obs_conc, penalize_ss, annealing_factor):
        return svi.update(state, obs_flux=obs_flux, obs_conc=obs_conc, penalize_ss=penalize_ss, annealing_factor=annealing_factor)

    progress_bar = tqdm(range(num_epochs), desc="Training", unit="epoch")
    for epoch in progress_bar:
        t = anneal(epoch, annealing_epochs, 0.1)
        state, loss = update_state(state, obs_flux=obs_flux, obs_conc=obs_conc, penalize_ss=penalize_ss, annealing_factor=t)
        progress_bar.set_postfix(loss=f"{loss:+.2e}", T=f"{t:.2f}")
    return maudy, svi.get_params(state)


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
    numpyro.enable_x64(True)
    maud_input = load_maud_input(str(maud_dir))
    maud_input._maudy_config = load_maudy_config(maud_dir)
    maudy, state = train(maud_input, num_epochs, penalize_ss, eval_flux, eval_conc, int(num_epochs * annealing_stage), normalize)
    if smoke:
        return
    var_names = ["y_flux_train", "latent_bal_conc", "unb_conc", "ssd", "dgr", "flux", "bal_conc"]
    samples = predict(maudy, state, 800, var_names)
    gathered_samples = report_to_dfs(maudy, samples, var_names=var_names)
    print_summary_dfs(gathered_samples)
    out = (
        out_dir
        if out_dir is not None
        else Path(f"maudyout_{maud_input.config.name}_{get_timestamp()}")
    )
    os.mkdir(out)
    with open(out / "maudy_result.pkl", "wb") as f:
        pickle.dump((maudy, state), f)
    shutil.copytree(maud_dir, out / "user_input")
