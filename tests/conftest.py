"""Fixtures for the test suite."""
from pathlib import Path
from pytest import fixture
from maudy.io import load_maudy_config
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
def maud_input(request):
    return load(Path(__file__).parent.parent / "examples" / request.param)
