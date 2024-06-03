from typer import Typer
from .train import sample
from .analysis import ppc


def main():
    app = Typer()
    # add sample and ppc functions as subcommands
    app.command("sample")(sample)
    app.command("ppc")(ppc)
    app()
