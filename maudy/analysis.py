"""Analyse the output of a model."""

from pathlib import Path

import pandas as pd
import torch
from maud.data_model.experiment import MeasurementType
from maud.loading_maud_inputs import load_maud_input
from pyro.optim import ClippedAdam
from pyro.infer import Predictive, config_enumerate

from .model import Maudy


def load(model_output: Path):
    user_input = str(model_output / "user_input")
    maud_input = load_maud_input(user_input)
    maudy = Maudy(maud_input)
    state_dict = torch.load(model_output / "model.pt", map_location="cpu")
    maudy.load_state_dict(state_dict["maudy"])
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


def predict(maudy: Maudy, num_epochs: int) -> tuple[pd.DataFrame, ...]:
    """Run posterior predictive check."""
    guide = config_enumerate(maudy.guide, "parallel", expand=True)
    predictive = Predictive(
        maudy.model,
        guide=guide,
        num_samples=num_epochs,
        return_sites=("y_flux_train", "steady_state_dev"),
    )
    samples = predictive()
    pred_summary = summary(samples)
    balanced_mics = [met.id for met in maudy.kinetic_model.mics if met.balanced]
    # assume obs_fluxes do not change across conditions
    obs_fluxes = [
        [
            meas.reaction
            for meas in exp.measurements
            if meas.target_type == MeasurementType.FLUX
        ]
        for exp in maudy.maud_params.experiments
    ][0]
    flux_measurements = {(exp.id, meas.reaction): meas.value for exp in maudy.maud_params.experiments for meas in exp.measurements if meas.target_type == MeasurementType.FLUX}
    mics_measurements = {(exp.id, "{meas.metabolite}_{meas.compartment}"): meas.value for exp in maudy.maud_params.experiments for meas in exp.measurements if meas.target_type == MeasurementType.MIC}
    ssds: list[pd.DataFrame] = []
    y_flux_trains: list[pd.DataFrame] = []
    for i, experiment in enumerate(maudy.experiments):
        ssd = {}
        y_flux_train = {}
        for key, items in pred_summary["steady_state_dev"].items():
            ssd[key] = items[i, :]
        for key, items in pred_summary["y_flux_train"].items():
            y_flux_train[key] = items[i]
        ssd = pd.DataFrame(ssd, index=balanced_mics)
        ssd["experiment"] = experiment
        ssds.append(ssd)
        y_flux_train = pd.DataFrame(y_flux_train, index=obs_fluxes)
        y_flux_train["experiment"] = experiment
        y_flux_trains.append(y_flux_train)

    ssd = pd.concat(ssds)
    y_flux_train = pd.concat(y_flux_trains)
    y_flux_train["measurement"] = y_flux_train.apply(lambda x: flux_measurements[(x["experiment"], x.name)] if (x["experiment"], x.name) in flux_measurements else None, axis=1)
    ssd["measurement"] = ssd.apply(lambda x: mics_measurements[(x["experiment"], x.name)] if (x["experiment"], x.name) in mics_measurements else None, axis=1)
    return y_flux_train, ssd


def ppc(model_output: Path, num_epochs: int = 800):
    maudy, _ = load(model_output)
    y_flux_train, ssd = predict(maudy, num_epochs)
    print("### Measured fluxes ###")
    print(y_flux_train)
    print("### Steady state dev ###")
    print(ssd)
