from typing import Optional

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
        """Initialize the priors of the model"""
        super().__init__()
        self.kinetic_model = maud_input.kinetic_model
        self.maud_params = maud_input.parameters
        # everything is sorted according to the reactions
        kcat_pars = self.maud_params.kcat.prior
        # we have to this splits because this maud is wrong...
        enzymatic_reactions = [x.split("_")[-1] for x in kcat_pars.ids[-1]]
        enzymes = [x.split("_")[0] for x in kcat_pars.ids[0]]
        kcats = pd.DataFrame(
            {"location": kcat_pars.location, "scale": kcat_pars.scale},
            index=enzymatic_reactions,
        )
        reactions = [reac.id for reac in self.kinetic_model.reactions]
        self.kcat_loc = torch.Tensor([
            kcats.loc[reac, "location"] for reac in enzymatic_reactions
        ])
        self.kcat_scale = torch.Tensor([
            kcats.loc[reac, "scale"] for reac in enzymatic_reactions
        ])
        ec = self.maud_params.conc_enzyme_train.prior
        enzyme_concs = pd.DataFrame(ec.location, index=ec.ids[0], columns=ec.ids[1])
        # sorted by reaction in the same order as kcats
        self.enzyme_concs_loc = torch.Tensor(enzyme_concs[enzymes].values)
        enzyme_concs = pd.DataFrame(ec.scale, index=ec.ids[0], columns=ec.ids[1])
        self.enzyme_concs_scale = torch.Tensor(enzyme_concs[enzymes].values)
        drain = self.maud_params.drain_train.prior
        drain_mean = pd.DataFrame(
            drain.location, index=drain.ids[0], columns=drain.ids[1]
        )
        self.drain_mean = torch.Tensor(drain_mean.values)
        self.drain_std = torch.Tensor(
            pd.DataFrame(drain.scale, index=drain.ids[0], columns=drain.ids[1]).values
        )

        self.experiments = ec.ids[0]
        self.num_reactions = len(reactions)
        mics = [met.id for met in self.kinetic_model.mics]
        edge_ids = drain.ids[1] + [
            f"{e}_{r}" for e, r in zip(enzymes, enzymatic_reactions)
        ]

        self.S = torch.FloatTensor(
            self.kinetic_model.stoichiometric_matrix.loc[mics, edge_ids].values
        )

        self.balanced_mics_idx = torch.LongTensor([
            i for i, met in enumerate(self.kinetic_model.mics) if met.balanced
        ])

        num_fluxes = len(ec.ids[1])
        self.obs_fluxes = torch.FloatTensor([
            [
                meas.value
                for meas in exp.measurements
                if meas.target_type == MeasurementType.FLUX
            ]
            for exp in self.maud_params.experiments
        ])
        self.obs_fluxes_std = torch.FloatTensor([
            [
                meas.error_scale
                for meas in exp.measurements
                if meas.target_type == MeasurementType.FLUX
            ]
            for exp in self.maud_params.experiments
        ])
        idx = torch.LongTensor([
            [
                enzymatic_reactions.index(meas.reaction)
                for meas in exp.measurements
                if meas.target_type == MeasurementType.FLUX
            ]
            for exp in self.maud_params.experiments
        ])
        self.num_obs_fluxes = len(idx[0])
        self.obs_fluxes_idx = (
            [i for i, l in enumerate(idx) for _ in l],
            [i for l in idx for i in l],
        )
        # Setup the various neural networks used in the model and guide
        self.odecoder = ToyDecoder(dims=[num_fluxes, 256, 256, self.num_obs_fluxes])

    def model(self, obs_fluxes: Optional[torch.FloatTensor] = None):
        """Describe the generative model."""
        # Register various nn.Modules (neural networks) with Pyro
        pyro.module("maudy", self)

        kcat = pyro.sample(
            "kcat", dist.LogNormal(self.kcat_loc, self.kcat_scale).to_event(1)
        )
        with pyro.plate("experiment", size=len(self.experiments), dim=-1):
            enzyme_conc = pyro.sample(
                "enzyme_conc",
                dist.LogNormal(self.enzyme_concs_loc, self.enzyme_concs_scale).to_event(
                    1
                ),
            )
            correction_scale = kcat.new_ones(self.num_obs_fluxes)
            correction = pyro.sample(
                "correction", dist.Normal(0, correction_scale).to_event(1)
            )
            # TODO: just a POC
            flux = pyro.deterministic("flux", get_vmax(kcat, enzyme_conc))
            true_obs_flux = flux[self.obs_fluxes_idx]
            drain = pyro.sample(
                "drain", dist.Normal(self.drain_mean, self.drain_std).to_event(1)
            )
            pyro.deterministic(
                "steady_state_dev",
                (torch.cat([drain, flux], dim=1) @ self.S.T)[:, self.balanced_mics_idx],
            )
            # Ensure true_obs_flux has shape [num_experiments, num_obs_fluxes]
            if true_obs_flux.ndim == 1:
                true_obs_flux = true_obs_flux.unsqueeze(-1)
            pyro.sample(
                "y_flux_train",
                dist.Normal(true_obs_flux * correction, self.obs_fluxes_std).to_event(
                    1
                ),
                obs=obs_fluxes,
            )

    # The guide specifies the variational distribution
    def guide(self, obs_fluxes: Optional[torch.FloatTensor] = None):
        """Establish the variational distributions for SVI."""
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
                dist.Normal(correction_loc, correction_scale).to_event(1),
            )
            steady_state_dev = pyro.sample(
                "steady_state_dev",
                dist.Normal(
                    correction_loc.new_zeros(len(self.balanced_mics_idx)), 1e-11
                ).to_event(1),
            )
            pyro.factor(
                "steady_state_loss", steady_state_dev.abs().sum(), has_rsample=True
            )

    def print_inputs(self):
        print(
            f"Exp: {self.experiments}\nNum reacs: {self.num_reactions}\n"
            f"Kcats: {self.kcat_loc};{self.kcat_scale}\n"
            f"Fluxes: {self.obs_fluxes};{self.obs_fluxes_std};{self.obs_fluxes_idx}"
        )

    def get_obs_fluxes(self):
        return self.obs_fluxes
