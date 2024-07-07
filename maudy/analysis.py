"""Analyse the output of a model."""

import dill as pickle
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import numpyro
import numpy as np
from jax import jit, random
import jax.numpy as jnp
from maud.data_model.experiment import MeasurementType
from maud.loading_maud_inputs import load_maud_input
from pyro.optim import ClippedAdam
from numpyro.infer import Predictive
from numpyro.infer.svi import SVIState


from .model import Maudy
from .io import load_maudy_config


def load(model_output: Path):
    with open(model_output / "maudy_result.pkl", "rb") as f:
        maudy, params = pickle.load(f)
    return maudy, params


def summary(samples):
    site_stats = {}
    for k, v in samples.items():
        site_stats[k] = {
            "mean": jnp.mean(v, axis=0),
            "std": jnp.std(v, axis=0),
            "5%": jnp.quantile(v, 0.05, axis=0),
            "95%": jnp.quantile(v, 0.95, axis=0),
        }
    return site_stats


def report_to_dfs(maudy: Maudy, samples: dict[Any, jnp.ndarray], var_names: list[str]):
    pred_summary = summary(samples)
    balanced_mics = [met.id for met in maudy.kinetic_model.mics if met.balanced]
    unbalanced_mics = [met.id for met in maudy.kinetic_model.mics if not met.balanced]
    # assume obs_fluxes do not change across conditions
    obs_fluxes = [
        [
            meas.reaction
            for meas in exp.measurements
            if meas.target_type == MeasurementType.FLUX
        ]
        for exp in maudy.maud_params.experiments
    ][0]
    flux_measurements = {
        (exp.id, meas.reaction): meas.value
        for exp in maudy.maud_params.experiments
        for meas in exp.measurements
        if meas.target_type == MeasurementType.FLUX
    }
    mics_measurements = {
        (exp.id, f"{meas.metabolite}_{meas.compartment}"): meas.value
        for exp in maudy.maud_params.experiments
        for meas in exp.measurements
        if meas.target_type == MeasurementType.MIC
    }
    kcat_pars = maudy.maud_params.kcat.prior
    enzymatic_reactions = [x.split("_")[-1] for x in kcat_pars.ids[-1]]
    across_exps = {var_name: [] for var_name in var_names}
    for i, experiment in enumerate(maudy.experiments):
        for var_name in across_exps.keys():
            this_dict = {}
            for key, items in pred_summary[var_name].items():
                this_dict[key] = items[i]
            df = pd.DataFrame(
                this_dict,
                index=obs_fluxes
                if var_name == "y_flux_train"
                else unbalanced_mics
                if "unb" in var_name
                else enzymatic_reactions
                if var_name in ["dgr", "flux"]
                else balanced_mics,
            )
            df["experiment"] = experiment
            across_exps[var_name].append(df)
    across_exps = {var_name: pd.concat(dfs) for var_name, dfs in across_exps.items()}
    for var_name in across_exps.keys():
        measurements = (
            flux_measurements if var_name == "y_flux_train" else mics_measurements
        )
        across_exps[var_name]["measurement"] = across_exps[var_name].apply(
            lambda x: measurements[(x["experiment"], x.name)]
            if (x["experiment"], x.name) in measurements
            else None,
            axis=1,
        )
        if across_exps[var_name]["measurement"].isnull().all():
            del across_exps[var_name]["measurement"]
    return across_exps


def predict(
    maudy: Maudy, svi_state: dict, num_samples: int, var_names: Sequence[str]
) -> dict[Any, jnp.ndarray]:
    """Run posterior predictive check."""
    maudy.encoder_args["train"] = False
    predictive = Predictive(
        maudy.model,
        guide=maudy.guide,
        params=svi_state,
        num_samples=num_samples,
        return_sites=list(var_names),
        parallel=False,  # the memory usage may be huge for large models if True
    )
    return jit(predictive)(rng_key=random.PRNGKey(23))

def print_summary_dfs(gathered_samples: dict[str, pd.DataFrame]):
    for var_name, df in gathered_samples.items():
        print(f"### {var_name} ###")
        if var_name == "dgr":
            df = df.loc[df.experiment == df.experiment.iloc[0], :]
            del df["experiment"]
        if var_name.startswith("ln_"):
            df.loc[:, ["mean", "5%", "95%"]] = np.exp(df[["mean", "5%", "95%"]])
        print(df) 


def ppc(model_output: Path, num_epochs: int = 800):
    """Run posterior predictive check and report it."""
    numpyro.enable_x64(True)
    var_names = (
        "y_flux_train",
        "latent_bal_conc",
        "unb_conc",
        "ssd",
        "dgr",
        "flux",
        "bal_conc",
    )
    maudy, params = load(model_output)
    samples = predict(maudy, params, num_epochs, var_names=var_names)
    gathered_samples = report_to_dfs(maudy, samples, var_names=list(var_names))
    print_summary_dfs(gathered_samples)
