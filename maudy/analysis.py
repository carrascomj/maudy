"""Analyse the output of a model."""

from pathlib import Path
from typing import Any

import pandas as pd
import pyro
import numpy as np
import torch
from maud.data_model.experiment import MeasurementType
from maud.loading_maud_inputs import load_maud_input
from pyro.optim import ClippedAdam
from pyro.infer import Predictive, config_enumerate

from .model import Maudy
from .io import load_maudy_config


def load(model_output: Path):
    user_input = model_output / "user_input"
    maud_input = load_maud_input(str(user_input))
    state_dict = torch.load(model_output / "model.pt", map_location="cpu")
    maud_input._maudy_config = load_maudy_config(user_input)
    # hack, if normalize was not applied, the decoder has a plain FF layer
    # otherwise, a sequential where the first element is the layer
    normalize = "decoder.loc_layer.0.bias" in state_dict["maudy"]
    quench = "quench.0.0.bias" in state_dict["maudy"]
    maudy = Maudy(maud_input, normalize, quench)
    maudy.load_state_dict(state_dict["maudy"])
    pyro.get_param_store().load(str(model_output / "model_params.pt"), map_location="cpu")
    optimizer = ClippedAdam({"lr": 0.006})
    optimizer.set_state(state_dict["optimizer"])
    return maudy, optimizer


def summary(samples):
    site_stats = {}
    for k, v in samples.items():
        site_stats[k] = {
            "mean": torch.mean(v, dim=0),
            "std": torch.std(v, dim=0),
            "5%": torch.quantile(v, 0.05, dim=0),
            "95%": torch.quantile(v, 0.95, dim=0),
        }
    return site_stats


def report_to_dfs(maudy: Maudy, samples: dict[Any, torch.Tensor], var_names: list[str]):
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
                this_dict[key] = items[i] if var_name != "dgr" else items.squeeze(0)
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
    maudy: Maudy, num_epochs: int, var_names: tuple[str, ...], oos: bool = False
) -> dict[Any, torch.Tensor]:
    """Run posterior predictive check."""
    maudy.concoder.set_dropout(0.0)
    guide = config_enumerate(maudy.guide, "parallel", expand=True)
    with torch.no_grad():
        return Predictive(
            maudy.model,
            guide=guide,
            num_samples=num_epochs,
            return_sites=var_names,
        )(None, None, False, 1.0, not oos)


def ppc(model_output: Path, num_epochs: int = 800):
    """Run posterior predictive check and report it."""
    var_names = (
        "y_flux_train",
        "latent_bal_conc",
        "unb_conc",
        "ssd",
        "dgr",
        "flux",
        "ln_bal_conc",
        "quench_correction",
    )
    maudy, _ = load(model_output)
    samples = predict(maudy, num_epochs, var_names=var_names)
    samples["ssd"] = samples["ssd"].squeeze(1)
    if "flux" in samples:
        samples["flux"] = samples["flux"].squeeze(1)
    gathered_samples = report_to_dfs(maudy, samples, var_names=list(var_names))
    for var_name, df in gathered_samples.items():
        print(f"### {var_name} ###")
        if var_name == "dgr":
            df = df.loc[df.experiment == df.experiment.iloc[0], :]
            del df["experiment"]
        if var_name.startswith("ln_"):
            df.loc[:, ["mean", "5%", "95%"]] = np.exp(df[["mean", "5%", "95%"]])
        print(df)
