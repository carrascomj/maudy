from itertools import product
from pathlib import Path

import pandas as pd
import torch
from maud.data_model.experiment import MeasurementType
from typer import Typer
from .train import sample
from .analysis import ppc, load
from .control import mca


def main():
    app = Typer()
    # add sample and ppc functions as subcommands
    app.command("sample")(sample)
    app.command("ppc")(ppc)
    app.command("control")(control)
    app()


def tensor_to_tidy_df(
    x: torch.Tensor, experiment_names: list[str], a_names: list[str], b_names: list[str]
) -> pd.DataFrame:
    """Convert a 4‑D tensor [Sample, Experiment, A, B] to a tidy DataFrame with
    columns [Sample, Experiment, A, B, Value].

    * Sample   -> int index of the first dimension
    * Experiment -> name from `experiment_names` by index
    * A -> name from `a_names` by index
    * B -> name from `b_names` by index
    * Value -> float value in the tensor
    """
    S, E, A, B = x.shape
    assert len(experiment_names) == E, "experiment_names length mismatch"
    assert len(a_names) == A, "a_names length mismatch"
    assert len(b_names) == B, "b_names length mismatch"

    # rows will be (s, e, a, b) for every possible combination
    index_tuples = list(product(range(S), range(E), range(A), range(B)))
    df = pd.DataFrame(
        index_tuples, columns=["Sample", "Experiment_idx", "A_idx", "B_idx"]
    )

    df["Experiment"] = df["Experiment_idx"].map(dict(enumerate(experiment_names)))
    df["A"] = df["A_idx"].map(dict(enumerate(a_names)))
    df["B"] = df["B_idx"].map(dict(enumerate(b_names)))

    df["Value"] = x.reshape(-1).detach().cpu().numpy().astype(float)

    df = df.drop(columns=["Experiment_idx", "A_idx", "B_idx"])
    df = df.astype(
        {
            "Sample": int,
            "Experiment": "string",
            "A": "string",
            "B": "string",
            "Value": float,
        }
    )

    df = df[["Sample", "Experiment", "A", "B", "Value"]]
    return df


def control(
    model_output: Path,
    prior_wrt: str = "enzyme_conc",
    d_conc: bool = True,
    num_epochs: int = 800,
):
    model, _ = load(model_output)
    model.to_double()
    C_V = mca(model, prior_wrt, d_conc=d_conc, samples=num_epochs)
    a_axis = "bal_conc" if d_conc else "flux"
    obs_fluxes = [
        [
            meas.reaction
            for meas in exp.measurements
            if meas.target_type == MeasurementType.FLUX
        ]
        for exp in model.maud_params.experiments
    ][0]

    balanced_mics = [str(met.id) for met in model.kinetic_model.mics if met.balanced]
    a_names = balanced_mics if d_conc else obs_fluxes
    kcat_pars = model.maud_params.kcat.prior
    enzymatic_reactions = [x.split("_")[-1] for x in kcat_pars.ids[-1]]
    b_names = (
        enzymatic_reactions
        if prior_wrt == "enzyme_conc"
        else [str(x) for x in range(C_V.shape[3])]
    )
    df = tensor_to_tidy_df(C_V, model.experiments, a_names, b_names)
    df = df.rename(columns={"A": a_axis, "B": prior_wrt})
    df.to_csv(model_output / "mca.tsv", sep="\t")
