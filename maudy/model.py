from copy import deepcopy
from typing import Optional

import pandas as pd
import pyro
import pyro.distributions as dist
import torch
import torch.nn as nn
from maud.data_model.maud_input import MaudInput
from maud.data_model.experiment import MeasurementType
from .black_box import ConcCoder
from .kinetics import (
    get_dgr,
    get_free_enzyme_ratio,
    get_kinetic_drain,
    get_reversibility,
    get_saturation,
    get_vmax,
)


class Maudy(nn.Module):
    def __init__(self, maud_input: MaudInput):
        """Initialize the priors of the model"""
        super().__init__()
        self.kinetic_model = maud_input.kinetic_model
        self.maud_params = maud_input.parameters
        # 1. kcats
        kcat_pars = self.maud_params.kcat.prior
        # we have to this splits because this maud is wrong...
        enzymatic_reactions = [x.split("_")[-1] for x in kcat_pars.ids[-1]]
        enzymes = [x.split("_")[0] for x in kcat_pars.ids[0]]
        mics = [met.id for met in self.kinetic_model.mics]
        mets = [met.id for met in self.kinetic_model.metabolites]
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
        # 2. enzyme concentrations
        ec = self.maud_params.conc_enzyme_train.prior
        enzyme_concs = pd.DataFrame(ec.location, index=ec.ids[0], columns=ec.ids[1])
        # sorted by reaction in the same order as kcats
        self.enzyme_concs_loc = torch.Tensor(enzyme_concs[enzymes].values)
        enzyme_concs = pd.DataFrame(ec.scale, index=ec.ids[0], columns=ec.ids[1])
        self.enzyme_concs_scale = torch.Tensor(enzyme_concs[enzymes].values)
        # 3. drains, fluxes are then concatenated in the model as drains + enzymatic_reations
        drain = self.maud_params.drain_train.prior
        drain_mean = pd.DataFrame(
            drain.location, index=drain.ids[0], columns=drain.ids[1]
        )
        self.drain_mean = torch.Tensor(drain_mean.values)
        self.drain_std = torch.Tensor(
            pd.DataFrame(drain.scale, index=drain.ids[0], columns=drain.ids[1]).values
        )
        # 4. dgfs
        dgf = self.maud_params.dgf.prior
        self.dgf_means = torch.Tensor(
            pd.Series(dgf.location, index=dgf.ids[0]).loc[mets]
        )
        dgf_cov = torch.Tensor(
            pd.DataFrame(dgf.covariance_matrix, index=dgf.ids[0], columns=dgf.ids[0])
            .loc[mets, mets]
            .values
        )
        self.dgf_cov = torch.linalg.cholesky(dgf_cov)
        self.experiments = ec.ids[0]
        self.num_reactions = len(reactions)
        self.num_mics = len(mics)
        edge_ids = drain.ids[1] + [
            f"{e}_{r}" for e, r in zip(enzymes, enzymatic_reactions)
        ]
        unb_conc = self.maud_params.conc_unbalanced_train.prior
        self.balanced_mics_idx = torch.LongTensor([
            i for i, met in enumerate(self.kinetic_model.mics) if met.balanced
        ])
        self.unbalanced_mics_idx = torch.LongTensor([
            i for i, met in enumerate(self.kinetic_model.mics) if not met.balanced
        ])
        unb_mics = [mic for i, mic in enumerate(mics) if i in self.unbalanced_mics_idx]
        bal_mics = [mic for i, mic in enumerate(mics) if i in self.balanced_mics_idx]
        self.unb_conc_loc = torch.FloatTensor(
            pd.DataFrame(
                unb_conc.location, index=unb_conc.ids[0], columns=unb_conc.ids[1]
            )
            .loc[self.experiments, unb_mics]
            .values
        )
        self.unb_conc_scale = torch.FloatTensor(
            pd.DataFrame(unb_conc.scale, index=unb_conc.ids[0], columns=unb_conc.ids[1])
            .loc[self.experiments, unb_mics]
            .values
        )
        conc_obs = {
            (exp.id, f"{meas.metabolite}_{meas.compartment}"): (
                meas.value,
                meas.error_scale,
            )
            for exp in self.maud_params.experiments
            for meas in exp.measurements
            if meas.target_type == MeasurementType.MIC
        }
        conc_inits = deepcopy(conc_obs)
        for exp in self.maud_params.experiments:
            for meas in exp.initial_state:
                conc_inits[(exp.id, meas.target_id)] = (meas.value, 1.0)
        self.bal_conc_loc = torch.FloatTensor([
            [
                conc_inits[(exp, mic)][0] if (exp, mic) in conc_inits else 1e-6
                for mic in bal_mics
            ]
            for exp in self.experiments
        ])
        self.bal_conc_scale = torch.FloatTensor([
            [
                conc_inits[(exp, mic)][1] if (exp, mic) in conc_inits else 1.0
                for mic in bal_mics
            ]
            for exp in self.experiments
        ])
        S = self.kinetic_model.stoichiometric_matrix.loc[mics, edge_ids]
        self.S = torch.FloatTensor(S.values)
        self.S_enz = torch.FloatTensor(S.loc[:, ~S.columns.isin(drain.ids[1])].values)
        mic_enz = S.loc[:, ~S.columns.isin(drain.ids[1])].index
        self.met_to_mic = torch.LongTensor([
            mets.index(mic.split("_", 1)[0]) for mic in mic_enz
        ])
        water_and_trans = {
            reac.id: (reac.water_stoichiometry, reac.transported_charge)
            for reac in self.kinetic_model.reactions
        }
        self.water_stoichiometry = torch.FloatTensor([
            water_and_trans[r][0] for r in enzymatic_reactions
        ])
        self.transported_charge = torch.FloatTensor([
            water_and_trans[r][1] for r in enzymatic_reactions
        ])
        # 5. saturation, we need kms and indices
        reac_st = {reac.id: reac.stoichiometry for reac in self.kinetic_model.reactions}
        self.sub_conc_idx = [
            torch.LongTensor([
                mics.index(met) for met, st in reac_st[reac].items() if st < 0
            ])
            for reac in enzymatic_reactions
        ]
        # the same but for drains
        assert all(st < 0 for reac in drain.ids[1] for _, st in reac_st[reac].items()), "drains are not implemented for products"
        self.sub_conc_drain_idx = [
            torch.LongTensor([
                mics.index(met) for met, st in reac_st[reac].items() if st < 0
            ])
            for reac in drain.ids[1]
        ]
        self.prod_conc_idx = [
            torch.LongTensor([
                mics.index(met) for met, st in reac_st[reac].items() if st > 0
            ])
            for reac in enzymatic_reactions
        ]
        self.substrate_S = [
            torch.LongTensor([-st for _, st in reac_st[reac].items() if st < 0])
            for reac in enzymatic_reactions
        ]
        self.product_S = [
            torch.LongTensor([st for _, st in reac_st[reac].items() if st > 0])
            for reac in enzymatic_reactions
        ]
        # the kms
        kms = self.maud_params.km.prior
        km_map = {}
        for i, km_id in enumerate(kms.ids[0]):
            enzyme, mic = km_id.split("_", 1)
            km_map[(enzyme, mic)] = i
        self.km_loc = torch.FloatTensor(kms.location)
        self.km_scale = torch.FloatTensor(kms.scale)
        self.sub_km_idx = [
            torch.LongTensor([
                km_map[(enz, met)] for met, st in reac_st[reac].items() if st < 0
            ])
            for enz, reac in zip(enzymes, enzymatic_reactions)
        ]
        # if the enz, mic is not in km_map, it is a irreversible reaction
        self.prod_km_idx = [
            torch.LongTensor([
                km_map[(enz, met)]
                for met, st in reac_st[reac].items()
                if st > 0 and (enz, met) in km_map
            ])
            for enz, reac in zip(enzymes, enzymatic_reactions)
        ]
        # verify that the indices are at least injective
        all_sub_idx = [
            conc_idx for reac_idx in self.sub_km_idx for conc_idx in reac_idx
        ]
        all_prod_idx = [
            conc_idx for reac_idx in self.prod_km_idx for conc_idx in reac_idx
        ]
        assert len(all_sub_idx) == len(
            set(all_sub_idx)
        ), "The indexing on the Km values went wrong for the substrates"
        assert len(all_prod_idx) == len(
            set(all_prod_idx)
        ), "The indexing on the Km values went wrong for the products"

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
        self.obs_conc = torch.FloatTensor([
            [conc_obs[(e, mic)][0] for mic in bal_mics if (e, mic) in conc_obs]
            for e in self.experiments
        ])
        self.obs_conc_std = torch.FloatTensor([
            [conc_obs[(e, mic)][1] for mic in bal_mics if (e, mic) in conc_obs]
            for e in self.experiments
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
            [i for i, exp in enumerate(idx) for _ in exp],
            [i for exp in idx for i in exp],
        )
        idx = torch.LongTensor([
            [
                bal_mics.index(f"{meas.metabolite}_{meas.compartment}")
                for meas in exp.measurements
                if meas.target_type == MeasurementType.MIC
            ]
            for exp in self.maud_params.experiments
        ])
        self.obs_conc_idx = (
            [i for i, exp in enumerate(idx) for _ in exp],
            [i for exp in idx for i in exp],
        )
        # Setup the various neural networks used in the model and guide
        self.concoder = ConcCoder(
            met_dims=[
                len(self.unbalanced_mics_idx) + len(self.balanced_mics_idx),
                16,
                16,
                len(self.balanced_mics_idx),
            ],
        )

    def model(
        self,
        obs_fluxes: Optional[torch.FloatTensor] = None,
        obs_conc: Optional[torch.FloatTensor] = None,
        penalize_ss: bool = True,
    ):
        """Describe the generative model."""
        # Register various nn.Modules (neural networks) with Pyro
        pyro.module("maudy", self)

        # experiment-indepedent variables
        kcat = pyro.sample(
            "kcat", dist.LogNormal(self.kcat_loc, self.kcat_scale).to_event(1)
        )
        dgf = pyro.sample(
            "dgf", dist.MultivariateNormal(self.dgf_means, scale_tril=self.dgf_cov)
        )
        dgf = dgf.reshape(-1)
        dgr = pyro.deterministic(
            "dgr", get_dgr(self.S_enz, dgf[self.met_to_mic], self.water_stoichiometry)
        )
        km = pyro.sample("km", dist.LogNormal(self.km_loc, self.km_scale).to_event(1))

        exp_plate = pyro.plate("experiment", size=len(self.experiments), dim=-1)
        with exp_plate:
            enzyme_conc = pyro.sample(
                "enzyme_conc",
                dist.LogNormal(self.enzyme_concs_loc, self.enzyme_concs_scale).to_event(
                    1
                ),
            )
            unb_conc = pyro.sample(
                "unb_conc",
                dist.LogNormal(self.unb_conc_loc, self.unb_conc_scale).to_event(1),
            )
            kcat_drain = (
                pyro.sample(
                    "kcat_drain", dist.Normal(self.drain_mean, self.drain_std).to_event(1)
                )
                if self.drain_mean.shape[1]
                else None
            )
            # TODO: need to take this from the config
            psi = pyro.sample("psi", dist.Normal(-0.110, 0.01))
            bal_conc = pyro.sample(
                "bal_conc",
                dist.LogNormal(self.bal_conc_loc.log(), self.bal_conc_scale).to_event(
                    1
                ),
            )
            correction = pyro.sample(
                "conc_correction",
                dist.LogNormal(
                    torch.zeros_like(self.bal_conc_loc),
                    0.1 * torch.ones_like(self.bal_conc_loc),
                ).to_event(1),
            )
            conc = kcat.new_ones(len(self.experiments), self.num_mics)
            conc[:, self.balanced_mics_idx] = bal_conc * correction
            conc[:, self.unbalanced_mics_idx] = unb_conc

        free_enzyme_ratio = pyro.deterministic(
            "free_enzyme_ratio",
            get_free_enzyme_ratio(
                conc,
                km,
                self.sub_conc_idx,
                self.sub_km_idx,
                self.prod_conc_idx,
                self.prod_km_idx,
                self.substrate_S,
                self.product_S,
            ),
        )
        vmax = pyro.deterministic("vmax", get_vmax(kcat, enzyme_conc))
        rev = pyro.deterministic(
            "rev",
            get_reversibility(self.S_enz, dgr, conc, self.transported_charge, psi),
        )
        sat = pyro.deterministic(
            "sat",
            get_saturation(
                conc, km, free_enzyme_ratio, self.sub_conc_idx, self.sub_km_idx
            ),
        )
        flux = pyro.deterministic("flux", vmax * rev * sat).reshape(
            len(self.experiments), len(self.sub_km_idx)
        )
        true_obs_flux = flux[self.obs_fluxes_idx]
        true_obs_conc = bal_conc[self.obs_conc_idx]
        # Ensure true_obs_flux has shape [num_experiments, num_obs_fluxes]
        true_obs_flux = true_obs_flux.reshape(len(self.experiments), -1)
        true_obs_conc = true_obs_conc.reshape(len(self.experiments), -1)
        if true_obs_flux.ndim == 1:
            true_obs_flux = true_obs_flux.unsqueeze(-1)
        if true_obs_conc.ndim == 1:
            true_obs_conc = true_obs_conc.unsqueeze(-1)

        # TODO: small drain correction from config
        drain = pyro.deterministic("drain", get_kinetic_drain(kcat_drain, conc, self.sub_conc_drain_idx, 1e-9)) if kcat_drain is not None else None
        all_flux = (
            torch.cat([drain, flux], dim=1) if drain is not None else flux
        ).clamp(-1e14, 1)
        ssd = pyro.deterministic("ssd", all_flux @ self.S.T[:, self.balanced_mics_idx])
        with exp_plate:
            pyro.sample(
                "y_flux_train",
                dist.Normal(true_obs_flux, self.obs_fluxes_std).to_event(1),
                obs=obs_fluxes,
            )
            pyro.sample(
                "y_conc_train",
                dist.LogNormal(true_obs_conc.log(), self.obs_conc_std).to_event(1),
                obs=obs_conc,
            )
            # steady state penalization
            pyro.sample(
                "steady_state_dev",
                dist.Normal(ssd, 10 * self.bal_conc_loc).to_event(1),
                obs=torch.zeros((len(self.experiments), len(self.balanced_mics_idx)))
                if penalize_ss
                else None,
            )

    # The guide specifies the variational distribution
    def guide(
        self,
        obs_fluxes: Optional[torch.FloatTensor] = None,
        obs_conc: Optional[torch.FloatTensor] = None,
        penalize_ss: bool = True,
    ):
        """Establish the variational distributions for SVI."""
        pyro.module("maudy", self)
        pyro.sample(
            "dgf", dist.MultivariateNormal(self.dgf_means, scale_tril=self.dgf_cov)
        )
        kcat = pyro.sample(
            "kcat", dist.LogNormal(self.kcat_loc, self.kcat_scale).to_event(1)
        )
        _ = pyro.sample("km", dist.LogNormal(self.km_loc, self.km_scale).to_event(1))
        exp_plate = pyro.plate("experiment", size=len(self.experiments), dim=-1)
        with exp_plate:
            _ = pyro.sample(
                "enzyme_conc",
                dist.LogNormal(self.enzyme_concs_loc, self.enzyme_concs_scale).to_event(
                    1
                ),
            )
            unb_conc = pyro.sample(
                "unb_conc",
                dist.LogNormal(self.unb_conc_loc, self.unb_conc_scale).to_event(1),
            )

            conc = kcat.new_ones(len(self.experiments), self.num_mics)
            bal_conc = pyro.sample(
                "bal_conc",
                dist.LogNormal(self.bal_conc_loc.log(), self.bal_conc_scale).to_event(
                    1
                ),
            )
            conc[:, self.balanced_mics_idx] = bal_conc
            conc[:, self.unbalanced_mics_idx] = unb_conc
            correction_loc = self.concoder(conc)
            pyro.sample(
                "conc_correction", dist.LogNormal(correction_loc, 0.1).to_event(1)
            )

            pyro.sample("psi", dist.Normal(-0.110, 0.01))
            _ = (
                pyro.sample(
                    "kcat_drain", dist.Normal(self.drain_mean, self.drain_std).to_event(1)
                )
                if self.drain_mean.shape[1]
                else None
            )

    def print_inputs(self):
        print(
            f"Exp: {self.experiments}\nNum reacs: {self.num_reactions}\n"
            f"Kcats: {self.kcat_loc};{self.kcat_scale}\n"
            f"Fluxes: {self.obs_fluxes};{self.obs_fluxes_std};{self.obs_fluxes_idx}"
        )

    def get_obs(self):
        return self.obs_fluxes, self.obs_conc
