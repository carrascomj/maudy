"""Run predictions."""

from pathlib import Path

import numpy as np
from maud.loading_maud_inputs import load_maud_input

from .analysis import load, predict, report_to_dfs
from .io import load_maudy_config
from .model import Maudy


def load_oos_model(trained_results: Path, oos_path: Path) -> Maudy:
    """Load the OOS Maudy object with the parameters of `trained_results`."""
    trained_maudy, _ = load(trained_results)
    state_dict = trained_maudy.state_dict()
    normalize = "decoder.loc_layer.0.bias" in state_dict
    quench = "quench.0.0.bias" in state_dict
    maud_input = load_maud_input(str(oos_path))
    maud_input._maudy_config = load_maudy_config(oos_path)
    print(f"{normalize=}; {quench=}")
    oos_maudy = Maudy(maud_input, normalize, quench)
    trained_maudy.experiments = oos_maudy.experiments
    trained_maudy.enzyme_concs_loc = oos_maudy.enzyme_concs_loc
    trained_maudy.enzyme_concs_scale = oos_maudy.enzyme_concs_scale
    trained_maudy.unb_conc_loc = oos_maudy.unb_conc_loc
    trained_maudy.unb_conc_scale = oos_maudy.unb_conc_scale
    trained_maudy.bal_conc_mu = oos_maudy.bal_conc_mu
    trained_maudy.bal_conc_loc = oos_maudy.bal_conc_loc
    trained_maudy.bal_conc_scale = oos_maudy.bal_conc_scale
    trained_maudy.obs_conc_mask = oos_maudy.obs_conc_mask
    trained_maudy.obs_conc_std = oos_maudy.obs_conc_std
    trained_maudy.obs_conc = oos_maudy.obs_conc
    trained_maudy.obs_fluxes_idx = oos_maudy.obs_fluxes_idx
    trained_maudy.obs_fluxes = oos_maudy.obs_fluxes
    trained_maudy.obs_fluxes_std = oos_maudy.obs_fluxes_std
    trained_maudy.drain_mean = oos_maudy.drain_mean
    trained_maudy.drain_std = oos_maudy.drain_std
    trained_maudy.maud_params = oos_maudy.maud_params

    # strict=False since there are experiment-specific parameters
    # oos_maudy.load_state_dict(state_dict, strict=True)
    return trained_maudy


def oos(trained_output: Path, oos_model_path: Path, num_epochs: int = 800):
    """Run posterior predictive on out-of-sample conditions and report it."""
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
    maudy = load_oos_model(trained_output, oos_model_path)
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
