import pandas as pd
import pyro
import pyro.distributions as dist
import torch
import torch.nn as nn
from maud.data_model.maud_input import MaudInput
from maud.data_model.experiment import MeasurementType
from .black_box import ToyDecoder
from .kinetics import get_vmax


class Maudy(nn.Module):
    def __init__(self, maud_input: MaudInput):
        super().__init__()
        self.kinetic_model = maud_input.kinetic_model
        self.maud_params = maud_input.parameters
        # everything is sorted according to the reactions
        kcat_pars = self.maud_params.kcat.prior
        # we have to this splits because this maud is wrong...
        reactions = [x.split("_")[-1] for x in kcat_pars.ids[-1]]
        enzymes = [x.split("_")[0] for x in kcat_pars.ids[0]]
        kcats = pd.DataFrame(
            {"location": kcat_pars.location, "scale": kcat_pars.scale},
            index=reactions,
        )
        self.kcat_loc = torch.Tensor([kcats.loc[reac, "location"] for reac in reactions])
        self.kcat_scale = torch.Tensor([kcats.loc[reac, "scale"] for reac in reactions])
        ec = self.maud_params.conc_enzyme_train.prior
        enzyme_concs = pd.DataFrame(ec.location, index=ec.ids[0], columns=ec.ids[1])
        # sorted by reaction in the same order as kcats
        self.enzyme_concs_loc = torch.Tensor(enzyme_concs[enzymes].values)
        enzyme_concs = pd.DataFrame(ec.scale, index=ec.ids[0], columns=ec.ids[1])
        self.enzyme_concs_scale = torch.Tensor(enzyme_concs[enzymes].values)
        self.experiments = ec.ids[0]
        self.num_reactions = len(reactions)

        # Setup the various neural networks used in the model and guide
        num_fluxes = len(ec.ids[1])
        self.obs_fluxes = torch.FloatTensor(
            [
                [
                    meas.value
                    for meas in exp.measurements
                    if meas.target_type == MeasurementType.FLUX
                ]
                for exp in self.maud_params.experiments
            ]
        )
        self.obs_fluxes_std = torch.FloatTensor(
            [
                [
                    meas.error_scale
                    for meas in exp.measurements
                    if meas.target_type == MeasurementType.FLUX
                ]
                for exp in self.maud_params.experiments
            ]
        )
        idx = torch.LongTensor(
            [
                [
                    meas.value
                    for meas in exp.measurements
                    if meas.target_type == MeasurementType.FLUX
                ]
                for exp in self.maud_params.experiments
            ]
        )
        self.num_obs_fluxes = len(idx[0])
        self.obs_fluxes_idx = [i for i, l in enumerate(idx) for _ in l], [i for l in idx for i in l]
        self.odecoder = ToyDecoder(dims=[num_fluxes, 256, 256, self.num_obs_fluxes])

        self.epsilon = 0.006

    def model(self):
        # Register various nn.Modules (neural networks) with Pyro
        pyro.module("maudy", self)

        kcat = pyro.sample("kcat", dist.LogNormal(self.kcat_loc, self.kcat_scale).to_event(1))
        with pyro.plate("experiment", size=len(self.experiments), dim=-1):
            enzyme_conc = pyro.sample(
                "enzyme_conc",
                dist.LogNormal(self.enzyme_concs_loc, self.enzyme_concs_scale).to_event(
                    1
                ),
            )
            correction_scale = kcat.new_ones(self.num_obs_fluxes)
            correction = pyro.sample(
                "correction", dist.LogNormal(0, correction_scale).to_event(1)
            )
            # TODO: just a POC
            flux = pyro.deterministic("flux", get_vmax(kcat, enzyme_conc))
            true_obs_flux = flux[self.obs_fluxes_idx[0]]
            pyro.sample(
                "y_flux_train",
                dist.Normal(true_obs_flux * correction, self.obs_fluxes_std).to_event(1),
                obs=self.obs_fluxes,
            )

    # The guide specifies the variational distribution
    def guide(self):
        pyro.module("maudy", self)
        with pyro.plate("experiment", size=len(self.experiments), dim=-1):
            enzyme_conc = pyro.sample(
                "enzyme_conc",
                dist.LogNormal(self.enzyme_concs_loc, self.enzyme_concs_scale).to_event(
                    1
                ),
            )
            correction_loc, correction_scale = self.odecoder(enzyme_conc)
            pyro.sample(
                "correction",
                dist.LogNormal(correction_loc, correction_scale).to_event(1),
            )

    def print_inputs(self):
        print(f"Exp: {self.experiments}\nNum reacs: {self.num_reactions}\nKcats: {self.kcat_loc};{self.kcat_scale}\nFluxes: {self.obs_fluxes};{self.obs_fluxes_std};{self.obs_fluxes_idx}")
