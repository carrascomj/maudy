"""Fixtures for the test suite."""
from pathlib import Path
from pytest import fixture

import arviz as az
import pandas as pd
from maudy.io import load_maudy_config
from maud.data_model.experiment import MeasurementType
from maud.data_model.hardcoding import ID_SEPARATOR
from maud.loading_maud_inputs import MaudInput, load_maud_input


def load(path: Path) -> MaudInput:
    maud_input = load_maud_input(str(path))
    maud_input._maudy_config = load_maudy_config(path)
    return maud_input


@fixture
def linear_maud_model() -> MaudInput:
    linear_path = Path(__file__).parent.parent / "examples" / "linear"
    return load(linear_path)


@fixture
def example_ode_model() -> MaudInput:
    linear_path = Path(__file__).parent.parent / "examples" / "example_ode_allos"
    return load(linear_path)


@fixture
def ci_aord_model() -> MaudInput:
    linear_path = Path(__file__).parent.parent / "examples" / "ci_aord_quench"
    return load(linear_path)


@fixture(params=["linear", "example_ode_allos"])
def maud_input(request) -> MaudInput:
    return load(Path(__file__).parent.parent / "examples" / request.param)


@fixture
def methionine_model() -> MaudInput:
    return load(Path(__file__).parent / "data" / "methionine_maud_output" / "user_input")


@fixture
def idata_methionine(methionine_model) -> az.InferenceData:
    mi = methionine_model
    mode = "train"
    experiments = (
        [e for e in mi.experiments if e.is_train]
        if mode == "train"
        else [e for e in mi.experiments if e.is_test]
    )
    yconc_coords, yflux_coords, yenz_coords = (
        [
            f"{e.id}{ID_SEPARATOR}{m.target_id}"
            for e in experiments
            for m in e.measurements
            if m.target_type == t
        ]
        for t in [
            MeasurementType.MIC,
            MeasurementType.FLUX,
            MeasurementType.ENZYME,
        ]
    )
    coords = {
        "enzymes": [e.id for e in mi.kinetic_model.enzymes],
        "experiments": [e.id for e in experiments],
        "reactions": [r.id for r in mi.kinetic_model.reactions],
        "drains": [r.id for r in mi.kinetic_model.drains],
        "metabolites": [m.id for m in mi.kinetic_model.metabolites],
        "mics": [m.id for m in mi.kinetic_model.mics],
        "edges": [e.id for e in mi.kinetic_model.edges],
        "unbalanced_mics": [
            m.id for m in mi.kinetic_model.mics if not m.balanced
        ],
        "balanced_mics": [m.id for m in mi.kinetic_model.mics if m.balanced],
        "phosphorylations": [p.id for p in mi.kinetic_model.phosphorylations]
        if mi.kinetic_model.phosphorylations is not None
        else [],
        "phosphorylation_modifying_enzymes": [
            pme.id for pme in mi.kinetic_model.phosphorylation_modifying_enzymes
        ]
        if mi.kinetic_model.phosphorylation_modifying_enzymes is not None
        else [],
        "allosteries": [p.id for p in mi.kinetic_model.allosteries]
        if mi.kinetic_model.allosteries is not None
        else [],
        "allosteric_enzymes": [
            e.id for e in mi.kinetic_model.allosteric_enzymes
        ]
        if mi.kinetic_model.allosteric_enzymes is not None
        else [],
        "competitive_inhibitions": [
            p.id for p in mi.kinetic_model.competitive_inhibitions
        ]
        if mi.kinetic_model.competitive_inhibitions is not None
        else [],
        "kms": mi.parameters.km.ids[0],
        "kis": mi.parameters.ki.ids[0],
        "dissociation_constants": (mi.parameters.dissociation_constant.ids[0]),
        "yconcs": yconc_coords,
        "yfluxs": yflux_coords,
        "yenz": yenz_coords,
    }
    dims = {
        f"flux_{mode}": ["experiments", "reactions"],
        f"conc_{mode}": ["experiments", "mics"],
        f"log_conc_enzyme_{mode}_z": ["experiments", "enzymes"],
        f"conc_enzyme_{mode}": ["experiments", "enzymes"],
        f"conc_unbalanced_{mode}": ["experiments", "unbalanced_mics"],
        f"conc_pme_{mode}": [
            "experiments",
            "phosphorylation_modifying_enzymes",
        ],
        f"drain_{mode}": ["experiments", "drains"],
        f"psi_{mode}": ["experiments"],
        f"saturation_{mode}": ["experiments", "edges"],
        f"free_enzyme_ratio_{mode}": ["experiments", "edges"],
        f"allostery_{mode}": ["experiments", "edges"],
        f"phosphorylation_{mode}": ["experiments", "edges"],
        f"reversibility_{mode}": ["experiments", "edges"],
        "dissociation_constant": ["allosteries"],
        "log_transfer_constant_z": ["allosteric_enzymes"],
        "transfer_constant": ["allosteric_enzymes"],
        "dgf": ["metabolites"],
        "dgr_train": ["experiments", "edges"],
        "keq": ["experiments", "edges"],
        "kcat": ["enzymes"],
        "kcat_pme": ["phosphorylation_modifying_enzymes"],
        "km": ["kms"],
        "ki": ["kis"],
        f"yrep_conc_{mode}": ["yconcs"],
        f"yrep_flux_{mode}": ["yfluxs"],
        f"llik_conc_{mode}": ["yconcs"],
        f"llik_flux_{mode}": ["yfluxs"],
        "concentration_control_matrix": [
            "experiments",
            "balanced_mics",
            "edges",
        ],
        "flux_control_matrix": ["experiments", "edges", "edges"],
        "flux_response_coefficient": ["experiments", "edges", "enzymes"],
        "concentration_response_coefficient": [
            "experiments",
            "balanced_mics",
            "enzymes",
        ],
    }
    idata = az.from_cmdstan(str(Path(__file__).parent / "data" / "methionine_maud_output" / "samples" / "*csv"), coords=coords, dims=dims)
    return idata


@fixture
def methionine_allostery(idata_methionine: az.InferenceData) -> tuple[pd.DataFrame, ...]:
    conc = (idata_methionine.posterior["conc_train"]
        .mean(dim=["chain", "draw"])
        .to_series()
        .unstack()
        .T)
    km = idata_methionine.posterior["km"].mean(dim=["chain", "draw"]).to_series().T
    ki = idata_methionine.posterior["ki"].mean(dim=["chain", "draw"]).to_series().T
    fer = idata_methionine.posterior["free_enzyme_ratio_train"].mean(dim=["chain", "draw"]).to_series().unstack()
    expected_allostery = (idata_methionine.posterior["allostery_train"]
        .mean(dim=["chain", "draw"])
        .to_series()
        .unstack()
        .T
    )
    tc = (
        idata_methionine.posterior["transfer_constant"]
        .mean(dim=["chain", "draw"])
        .to_series()
    )
    dc = (
        idata_methionine.posterior["dissociation_constant"]
        .mean(dim=["chain", "draw"])
        .to_series()
    )
    saturation = (
        idata_methionine.posterior["saturation_train"]
        .mean(dim=["chain", "draw"])
        .to_series()
        .unstack()
        .T
    )
    kcat = (
        idata_methionine.posterior["kcat"]
        .mean(dim=["chain", "draw"])
        .to_series()
    )
    enzyme_conc = (
        idata_methionine.posterior["conc_enzyme_train"]
        .mean(dim=["chain", "draw"])
        .to_series()
        .unstack()
        .T
    )
    return conc, km, ki, fer, tc, dc, expected_allostery, saturation, kcat, enzyme_conc

@fixture
def methionine_reversibility(idata_methionine: az.InferenceData) -> tuple[pd.DataFrame, ...]:
    dgf = (
        idata_methionine.posterior["dgf"]
        .mean(dim=["chain", "draw"])
        .to_series()
        .T
    )
    dgr = (
        idata_methionine.posterior["dgr_train"]
        .mean(dim=["chain", "draw"])
        .to_series()
        .unstack()
        .T
    )
    reversibility = (
        idata_methionine.posterior["reversibility_train"]
        .mean(dim=["chain", "draw"])
        .to_series()
        .unstack()
        .T
    )
    psi = (
        idata_methionine.posterior["psi_train"]
        .mean(dim=["chain", "draw"])
        .to_series()
        .T
    )

    return dgf, dgr, reversibility, psi

