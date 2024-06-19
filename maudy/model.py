from collections import defaultdict
from copy import deepcopy
from typing import Optional

import pandas as pd
import pyro
import pyro.distributions as dist
import torch
import torch.nn as nn
from maud.data_model.maud_input import MaudInput
from maud.data_model.experiment import MeasurementType
from .black_box import BaseConcCoder, fdx_head, unb_opt_head
from .kinetics import (
    get_allostery,
    get_dgr,
    get_free_enzyme_ratio_denom,
    get_competitive_inhibition_denom,
    get_kinetic_multi_drain,
    get_reversibility,
    get_saturation,
    get_vmax,
)

Positive = dist.constraints.positive


def get_loc_from_mu_scale(mu: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    mu2 = mu.pow(2)
    sigma_sq = (scale.exp() - 1) * mu2
    loc = torch.log(mu.pow(2) / torch.sqrt(mu2 + sigma_sq.pow(2)))
    return loc


class Maudy(nn.Module):
    def __init__(self, maud_input: MaudInput):
        """Initialize the priors of the model.

        maud_input: MaudInput
            a `MaudInput` with an optionally injected `._fdx_stoichiometry`
            dict atttribute with identifiers as keys and its ferredoxin
            stoichiometry as values (defaults 0 for reactions not in the dict).
        optimize_unbalanced: Optional[list[str]]
            unbalanced metabolite-in-compartment identifiers to infer as the
            output of the neural network.
        """
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
        self.kcat_loc = torch.Tensor(
            [kcats.loc[reac, "location"] for reac in enzymatic_reactions]
        )
        self.kcat_scale = torch.Tensor(
            [kcats.loc[reac, "scale"] for reac in enzymatic_reactions]
        )
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
        self.balanced_mics_idx = torch.LongTensor(
            [i for i, met in enumerate(self.kinetic_model.mics) if met.balanced]
        )
        self.unbalanced_mics_idx = torch.LongTensor(
            [i for i, met in enumerate(self.kinetic_model.mics) if not met.balanced]
        )
        unb_mics = [mic for i, mic in enumerate(mics) if i in self.unbalanced_mics_idx]
        bal_mics = [mic for i, mic in enumerate(mics) if i in self.balanced_mics_idx]
        unb_config = maud_input._maudy_config.optimize_unbalanced_metabolites
        optimize_unbalanced = unb_config if unb_config is not None else []
        if optimize_unbalanced:
            opt_not_unb = set(optimize_unbalanced) - set(unb_mics)
            assert (
                not opt_not_unb
            ), f"{opt_not_unb} to be optimized are not unbalanced metabolites!"
        self.optimized_unbalanced_idx = torch.LongTensor(
            [i for i, met in enumerate(unb_mics) if met in optimize_unbalanced]
        )
        self.non_optimized_unbalanced_idx = torch.LongTensor(
            [i for i, met in enumerate(unb_mics) if met not in optimize_unbalanced]
        )
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
        self.bal_conc_mu = torch.FloatTensor(
            [
                [
                    conc_inits[(exp, mic)][0] if (exp, mic) in conc_inits else 1e-6
                    for mic in bal_mics
                ]
                for exp in self.experiments
            ]
        )
        self.bal_conc_scale = torch.FloatTensor(
            [
                [
                    conc_inits[(exp, mic)][1] if (exp, mic) in conc_inits else 1.0
                    for mic in bal_mics
                ]
                for exp in self.experiments
            ]
        )
        self.bal_conc_loc = get_loc_from_mu_scale(self.bal_conc_mu, self.bal_conc_scale)
        S = self.kinetic_model.stoichiometric_matrix.loc[mics, edge_ids]
        self.S = torch.FloatTensor(S.values)
        # S matrix only for stoichoimetric reactions (to calculate saturation and drG)
        self.S_enz = torch.FloatTensor(S.loc[:, ~S.columns.isin(drain.ids[1])].values)
        # S matrix used for thermo drG prime (reverisibility term), same as the one
        # for enzymatic reactions but it may be modified later if ferredoxin is present
        self.S_enz_thermo = torch.FloatTensor(
            S.loc[:, ~S.columns.isin(drain.ids[1])].values
        )
        mic_enz = S.loc[:, ~S.columns.isin(drain.ids[1])].index
        self.met_to_mic = torch.LongTensor(
            [mets.index(mic.split("_", 1)[0]) for mic in mic_enz]
        )
        water_and_trans = {
            reac.id: (reac.water_stoichiometry, reac.transported_charge)
            for reac in self.kinetic_model.reactions
        }
        self.water_stoichiometry = torch.FloatTensor(
            [water_and_trans[r][0] for r in enzymatic_reactions]
        )
        self.transported_charge = torch.FloatTensor(
            [water_and_trans[r][1] for r in enzymatic_reactions]
        )
        # 5. saturation, we need kms and indices
        reac_st = {reac.id: reac.stoichiometry for reac in self.kinetic_model.reactions}
        self.sub_conc_idx = [
            torch.LongTensor(
                [mics.index(met) for met, st in reac_st[reac].items() if st < 0]
            )
            for reac in enzymatic_reactions
        ]
        self.prod_conc_idx = [
            torch.LongTensor(
                [mics.index(met) for met, st in reac_st[reac].items() if st > 0]
            )
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
        # the same but for drains
        self.sub_conc_drain_idx = [
            torch.LongTensor(
                [mics.index(met) for met, st in reac_st[reac].items() if st < 0]
            )
            for reac in drain.ids[1]
        ]
        self.prod_conc_drain_idx = [
            torch.LongTensor(
                [mics.index(met) for met, st in reac_st[reac].items() if st > 0]
            )
            for reac in drain.ids[1]
        ]
        self.substrate_drain_S = [
            torch.LongTensor([-st for _, st in reac_st[reac].items() if st < 0])
            for reac in drain.ids[1]
        ]
        self.product_drain_S = [
            torch.LongTensor([st for _, st in reac_st[reac].items() if st > 0])
            for reac in drain.ids[1]
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
            torch.LongTensor(
                [km_map[(enz, met)] for met, st in reac_st[reac].items() if st < 0]
            )
            for enz, reac in zip(enzymes, enzymatic_reactions)
        ]
        # if the enz, mic is not in km_map, it is a irreversible reaction
        self.prod_km_idx = [
            torch.LongTensor(
                [
                    km_map[(enz, met)]
                    for met, st in reac_st[reac].items()
                    if st > 0 and (enz, met) in km_map
                ]
            )
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
        # competitive inhibition
        kis = self.maud_params.ki.prior
        self.has_ci = False
        if kis.location:
            self.has_ci = True
            ki_map = defaultdict(list)
            for i, ki_id in enumerate(kis.ids[0]):
                enzyme, _, mic = ki_id.split("_", 2)
                ki_map[enzyme].append((i, mics.index(mic)))
            self.ki_loc = torch.FloatTensor(kis.location)
            self.ki_scale = torch.FloatTensor(kis.scale)
            self.ki_idx = [
                torch.LongTensor([i_ki for i_ki, _ in ki_map[enz]]) for enz in enzymes
            ]
            self.ki_conc_idx = [
                torch.LongTensor(
                    [i_conc for _, i_conc in ki_map[enz]] if enz in ki_map else []
                )
                for enz in enzymes
            ]
        # allostery
        dc = self.maud_params.dissociation_constant.prior
        tc = self.maud_params.transfer_constant.prior
        self.has_allostery = False
        if dc.location:
            self.has_allostery = True
            dc_map = {}
            for dc_id in dc.ids[0]:
                enzyme, met, comp, atype = dc_id.split("_", 3)
                dc_map[enzyme] = (
                    enzymes.index(enzyme),
                    mics.index(f"{met}_{comp}"),
                    atype,
                )
            self.dc_loc = torch.FloatTensor(dc.location)
            self.dc_scale = torch.FloatTensor(dc.scale)
            self.tc_loc = torch.FloatTensor(tc.location)
            self.tc_scale = torch.FloatTensor(tc.scale)
            self.allostery_idx = torch.LongTensor(
                [dc_map[enz][0] for enz in enzymes if enz in dc_map]
            )
            self.conc_allostery_idx = torch.LongTensor(
                [dc_map[enz][1] for enz in enzymes if enz in dc_map]
            )
            self.allostery_activation = torch.BoolTensor(
                [dc_map[enz][2] == "activation" for enz in enzymes if enz in dc_map]
            )
            self.subunits = torch.IntTensor(
                [
                    next(
                        kin_enz.subunits
                        for kin_enz in self.kinetic_model.enzymes
                        if kin_enz.id == enz
                    )
                    for enz in enzymes
                    if enz in dc_map
                ]
            )

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
        self.obs_conc = torch.FloatTensor(
            [
                [conc_obs[(e, mic)][0] for mic in bal_mics if (e, mic) in conc_obs]
                for e in self.experiments
            ]
        )
        self.obs_conc_std = torch.FloatTensor(
            [
                [conc_obs[(e, mic)][1] for mic in bal_mics if (e, mic) in conc_obs]
                for e in self.experiments
            ]
        )
        idx = torch.LongTensor(
            [
                [
                    enzymatic_reactions.index(meas.reaction)
                    for meas in exp.measurements
                    if meas.target_type == MeasurementType.FLUX
                ]
                for exp in self.maud_params.experiments
            ]
        )
        self.num_obs_fluxes = len(idx[0])
        self.obs_fluxes_idx = (
            [i for i, exp in enumerate(idx) for _ in exp],
            [i for exp in idx for i in exp],
        )
        idx = torch.LongTensor(
            [
                [
                    bal_mics.index(f"{meas.metabolite}_{meas.compartment}")
                    for meas in exp.measurements
                    if meas.target_type == MeasurementType.MIC
                ]
                for exp in self.maud_params.experiments
            ]
        )
        self.obs_conc_idx = (
            [i for i, exp in enumerate(idx) for _ in exp],
            [i for exp in idx for i in exp],
        )
        # Special case of ferredoxin: we want to add a per-experiment
        # concentration ratio parameter (output of NN) and the dGf difference
        self.fdx_stoichiometry = torch.zeros_like(self.water_stoichiometry)
        fdx = maud_input._maudy_config.ferredoxin
        if fdx is not None:
            # first check if all reactions have a correct identifier to catch user typos
            fdx_not_reac = set(fdx.keys()) - set(enzymatic_reactions)
            assert (
                len(fdx_not_reac) == 0
            ), f"{fdx_not_reac} with ferredoxin not in {enzymatic_reactions}"
            self.fdx_stoichiometry = torch.FloatTensor(
                [fdx[r] if r in fdx else 0 for r in enzymatic_reactions]
            )
            # add row for S matrix to calculate DrG prime
            self.S_enz_thermo = torch.cat(
                [self.S_enz_thermo, self.fdx_stoichiometry.unsqueeze(0)], dim=0
            )
        self.has_fdx = any(st != 0 for st in self.fdx_stoichiometry)
        nn_config = maud_input._maudy_config.neural_network
        # Setup the various neural networks used in guide
        nn_decoder = BaseConcCoder(
            met_dims=[len(self.non_optimized_unbalanced_idx)]
            + nn_config.met_dims
            + [len(self.balanced_mics_idx)],
            reac_dims=[len(enzymatic_reactions)] + nn_config.reac_dims,
            km_dims=[len(self.km_loc)] + nn_config.km_dims,
            drain_dim=self.drain_mean.shape[1] if len(self.drain_mean.size()) else 0,
            ki_dim=self.ki_loc.shape[0] if hasattr(self, "ki_loc") else 0,
            tc_dim=self.tc_loc.shape[0] if hasattr(self, "tc_loc") else 0,
        )
        nn_encoder = BaseConcCoder(
            met_dims=[self.obs_conc.shape[-1]]
            + nn_config.met_dims
            + [len(self.balanced_mics_idx)],
            reac_dims=[len(enzymatic_reactions)] + nn_config.reac_dims,
            km_dims=[len(self.km_loc)] + nn_config.km_dims,
            drain_dim=self.drain_mean.shape[1] if len(self.drain_mean.size()) else 0,
            ki_dim=self.ki_loc.shape[0] if hasattr(self, "ki_loc") else 0,
            tc_dim=self.tc_loc.shape[0] if hasattr(self, "tc_loc") else 0,
            obs_flux_dim=self.obs_fluxes.shape[-1],
        )
        if self.has_fdx:
            fdx_head(nn_decoder)
            fdx_head(nn_encoder)
        if len(self.optimized_unbalanced_idx.size()) != 0:
            unb_opt_head(nn_decoder, unb_dim=self.optimized_unbalanced_idx.shape[-1])
            unb_opt_head(nn_encoder, unb_dim=self.optimized_unbalanced_idx.shape[-1])
        self.concoder = nn_decoder
        self.concdecoder = nn_encoder

    def cuda(self):
        super().cuda()
        for key in self.__dict__.keys():
            if isinstance(self.__dict__[key], torch.Tensor):
                self.__dict__[key] = self.__dict__[key].cuda()
            if isinstance(self.__dict__[key], list):
                self.__dict__[key] = [
                    x.cuda() if isinstance(x, torch.Tensor) else x
                    for x in self.__dict__[key]
                ]

    def model(
        self,
        obs_flux: Optional[torch.FloatTensor] = None,
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
        fdx_contr = (
            pyro.sample("fdx_contr", dist.Normal(77, 1))
            if self.has_fdx
            else self.float_tensor([0.0])
        )
        dgr = pyro.deterministic(
            "dgr",
            get_dgr(
                self.S_enz,
                dgf[self.met_to_mic],
                self.water_stoichiometry,
                self.fdx_stoichiometry,
                fdx_contr,
            ),
        )
        km = pyro.sample("km", dist.LogNormal(self.km_loc, self.km_scale).to_event(1))
        rest = self.float_tensor([])
        if self.has_ci:
            ki = pyro.sample(
                "ki", dist.LogNormal(self.ki_loc, self.ki_scale).to_event(1)
            )
            rest = ki
        if self.has_allostery:
            dc = pyro.sample(
                "dc", dist.LogNormal(self.dc_loc, self.dc_scale).to_event(1)
            )
            tc = pyro.sample(
                "tc", dist.LogNormal(self.tc_loc, self.tc_scale).to_event(1)
            )
            rest = torch.cat([rest, tc, dc], dim=-1)

        exp_plate = pyro.plate("experiment", size=len(self.experiments), dim=-1)
        with exp_plate:
            enzyme_conc = pyro.sample(
                "enzyme_conc",
                dist.LogNormal(self.enzyme_concs_loc, self.enzyme_concs_scale).to_event(
                    1
                ),
            )
            unb_conc_param_loc = pyro.param(
                "unb_conc_param_loc",
                self.unb_conc_loc[:, self.non_optimized_unbalanced_idx]
            )
            kcat_drain = (
                pyro.sample(
                    "kcat_drain",
                    dist.Normal(self.drain_mean, self.drain_std).to_event(1),
                )
                if self.drain_mean.shape[1]
                else self.float_tensor([])
            )
            # TODO: need to take this from the config
            psi = pyro.sample(
                "psi", dist.Normal(self.float_tensor(-0.110), self.float_tensor(0.01))
            )
            # there is a bug in Predict(parallel=True) that may add extra dims
            # and, thus, the squeezes in 1-dim variables
            concoder_output = self.concoder(
                unb_conc_param_loc,
                dgr,
                enzyme_conc,
                kcat.squeeze(0),
                kcat_drain,
                km.squeeze(0),
                rest.squeeze(0),
            )
            unb_conc_param_loc_full = torch.full_like(self.unb_conc_loc, 1.0)
            unb_conc_param_loc_full[:, self.non_optimized_unbalanced_idx] = (
                unb_conc_param_loc
            )
            # in reverse order that additional head outputs may have been added
            if len(self.optimized_unbalanced_idx.size()) != 0:
                unb_optimized = concoder_output.pop()
                unb_conc_param_loc_full[:, self.optimized_unbalanced_idx] = (
                    unb_optimized
                )
            if self.has_fdx:
                fdx_contr = concoder_output.pop()
            bal_conc_loc, bal_conc_scale = concoder_output
            unb_conc = pyro.sample(
                "unb_conc",
                dist.LogNormal(unb_conc_param_loc_full, self.unb_conc_scale).to_event(
                    1
                ),
            )
            latent_bal_conc = pyro.sample(
                "latent_bal_conc",
                dist.LogNormal(bal_conc_loc, bal_conc_scale).to_event(1),
            )
            pyro.deterministic("ln_bal_conc", bal_conc_loc)
            conc = kcat.new_ones(len(self.experiments), self.num_mics)
            conc[:, self.balanced_mics_idx] = latent_bal_conc
            conc[:, self.unbalanced_mics_idx] = unb_conc
            if self.has_fdx:
                fdx_ratio = pyro.sample(
                    "fdx_ratio", dist.LogNormal(fdx_contr, 0.1).to_event(1)
                )
                conc = torch.cat([conc, fdx_ratio], dim=1)
        # log balanced concentrations
        free_enz_km_denom = get_free_enzyme_ratio_denom(
            conc,
            km,
            self.sub_conc_idx,
            self.sub_km_idx,
            self.prod_conc_idx,
            self.prod_km_idx,
            self.substrate_S,
            self.product_S,
        )
        free_enz_ki_denom = (
            pyro.deterministic(
                "ci",
                get_competitive_inhibition_denom(
                    conc,
                    ki,
                    self.ki_conc_idx,
                    self.ki_idx,
                ),
            )
            if self.has_ci
            else 0
        )
        free_enzyme_ratio = pyro.deterministic(
            "free_enzyme_ratio",
            1 / (free_enz_km_denom + free_enz_ki_denom),
        )
        vmax = pyro.deterministic("vmax", get_vmax(kcat, enzyme_conc))
        rev = pyro.deterministic(
            "rev",
            get_reversibility(
                self.S_enz_thermo, dgr, conc, self.transported_charge, psi
            ),
        )
        sat = pyro.deterministic(
            "sat",
            get_saturation(
                conc, km, free_enzyme_ratio, self.sub_conc_idx, self.sub_km_idx
            ),
        )
        allostery = (
            pyro.deterministic(
                "allostery",
                get_allostery(
                    conc,
                    free_enzyme_ratio,
                    tc,
                    dc,
                    self.allostery_activation,
                    self.allostery_idx,
                    self.conc_allostery_idx,
                    self.subunits,
                ),
            )
            if self.has_allostery
            else torch.ones_like(vmax)
        )
        flux = pyro.deterministic("flux", vmax * rev * sat * allostery).reshape(
            len(self.experiments), len(self.sub_km_idx)
        )
        true_obs_flux = flux[self.obs_fluxes_idx]
        true_obs_conc = bal_conc_loc[self.obs_conc_idx]
        # Ensure true_obs_flux has shape [num_experiments, num_obs_fluxes]
        true_obs_flux = true_obs_flux.reshape(len(self.experiments), -1)
        true_obs_conc = true_obs_conc.reshape(len(self.experiments), -1)
        if true_obs_flux.ndim == 1:
            true_obs_flux = true_obs_flux.unsqueeze(-1)
        if true_obs_conc.ndim == 1:
            true_obs_conc = true_obs_conc.unsqueeze(-1)

        # TODO: small drain correction from config
        drain = (
            pyro.deterministic(
                "drain",
                get_kinetic_multi_drain(
                    kcat_drain,
                    conc,
                    self.sub_conc_drain_idx,
                    self.prod_conc_drain_idx,
                    self.substrate_drain_S,
                    self.product_drain_S,
                    1e-9,
                ),
            )
            if kcat_drain.size()[0]
            else None
        )
        all_flux = (
            torch.cat([drain, flux], dim=1) if drain is not None else flux
        ).clamp(-1e14, 100)
        ssd = pyro.deterministic(
            "ssd", all_flux @ self.S.T[:, self.balanced_mics_idx]
        )
        with exp_plate:
            pyro.sample(
                "y_flux_train",
                dist.Normal(true_obs_flux, self.obs_fluxes_std).to_event(1),
                obs=obs_flux,
            )
            pyro.sample(
                "y_conc_train",
                dist.LogNormal(true_obs_conc, self.obs_conc_std).to_event(1),
                obs=obs_conc,
            )

    def float_tensor(self, x) -> torch.Tensor:
        return torch.tensor(x, device=self.water_stoichiometry.device)

    # The guide specifies the variational distribution
    def guide(
        self,
        obs_flux: Optional[torch.FloatTensor] = None,
        obs_conc: Optional[torch.FloatTensor] = None,
        penalize_ss: bool = False,
    ):
        """Establish the variational distributions for SVI."""
        pyro.module("maudy", self)
        dgf_param_loc = pyro.param("dgf_loc", self.dgf_means)
        dgf = pyro.sample(
            "dgf", dist.MultivariateNormal(dgf_param_loc, scale_tril=self.dgf_cov)
        )
        fdx_contr_loc = pyro.param("fdx_contr_loc", self.float_tensor([77.0]))
        fdx_contr_scale = pyro.param(
            "fdx_contr_scale", self.float_tensor([1.0]), constraint=Positive
        )
        fdx_contr = (
            pyro.sample("fdx_contr", dist.Normal(fdx_contr_loc, fdx_contr_scale))
            if any(st != 0 for st in self.fdx_stoichiometry)
            else self.float_tensor([0.0])
        )
        kcat_param_loc = pyro.param("kcat_loc", self.kcat_loc)
        kcat = pyro.sample(
            "kcat", dist.LogNormal(kcat_param_loc, self.kcat_scale).to_event(1)
        )
        km_loc = pyro.param("km_loc", self.km_loc)
        km_scale = pyro.param("km_scale", self.km_scale, Positive)
        km = pyro.sample("km", dist.LogNormal(km_loc, km_scale).to_event(1))
        dgr = get_dgr(
            self.S_enz,
            dgf[self.met_to_mic],
            self.water_stoichiometry,
            self.fdx_stoichiometry,
            fdx_contr,
        )
        rest = self.float_tensor([])
        if self.has_ci:
            ki_loc = pyro.param("ki_loc", self.ki_loc)
            ki_scale = pyro.param("ki_scale", self.ki_scale, Positive)
            ki = pyro.sample("ki", dist.LogNormal(ki_loc, ki_scale).to_event(1))
            rest = ki
        if self.has_allostery:
            dc_loc = pyro.param("dc_loc", self.dc_loc)
            dc_scale = pyro.param("dc_scale", self.dc_scale, Positive)
            tc_loc = pyro.param("tc_loc", self.tc_loc)
            tc_scale = pyro.param("tc_scale", self.tc_scale, Positive)
            dc = pyro.sample("dc", dist.LogNormal(dc_loc, dc_scale).to_event(1))
            tc = pyro.sample("tc", dist.LogNormal(tc_loc, tc_scale).to_event(1))
            rest = torch.cat([rest, tc, dc])
        exp_plate = pyro.plate("experiment", size=len(self.experiments), dim=-1)
        with exp_plate:
            enzyme_concs_param_loc = pyro.param(
                "enzyme_concs_loc", self.enzyme_concs_loc, event_dim=1
            )
            enz_conc = pyro.sample(
                "enzyme_conc",
                dist.LogNormal(
                    enzyme_concs_param_loc, self.enzyme_concs_scale
                ).to_event(1),
            )
            psi_mean = pyro.param("psi_mean", self.float_tensor([-0.110]))
            psi = pyro.sample("psi", dist.Normal(psi_mean, 0.01))
            drain_mean = pyro.param("drain_mean", lambda: self.drain_mean)
            drain_std = pyro.param(
                "drain_std", lambda: self.drain_std, constraint=Positive
            )
            kcat_drain = (
                pyro.sample(
                    "kcat_drain",
                    dist.Normal(drain_mean, drain_std).to_event(1),
                )
                if self.drain_mean.shape[1]
                else self.float_tensor([])
            )
            # if inference, use the concoder
            if obs_conc is None and obs_flux is None:
                # prepare NN conc input, remains 1 for unbalanced optiimzed mets
                conc_nn_input = self.unb_conc_loc[:, self.non_optimized_unbalanced_idx]
                concoder_output = self.concoder(
                    conc_nn_input, dgr, enz_conc, kcat, kcat_drain, km, rest
                )
            else:
                concoder_output = self.concdecoder(
                    obs_conc, dgr, enz_conc, kcat, kcat_drain, km, rest, obs_flux
                )
            unb_conc_param_loc_full = torch.full_like(self.unb_conc_loc, 1.0)
            unb_conc_param_loc_full[:, self.non_optimized_unbalanced_idx] = (
                self.unb_conc_loc[:, self.non_optimized_unbalanced_idx]
            )
            # in reverse order that additional head outputs may have been added
            if len(self.optimized_unbalanced_idx.size()) != 0:
                unb_optimized = concoder_output.pop()
                unb_conc_param_loc_full[:, self.optimized_unbalanced_idx] = (
                    unb_optimized
                )
            if self.has_fdx:
                fdx_contr = concoder_output.pop()
            bal_conc_loc, bal_conc_scale = concoder_output
            unb_conc = pyro.sample(
                "unb_conc",
                dist.LogNormal(unb_conc_param_loc_full, self.unb_conc_scale).to_event(
                    1
                ),
            )
            latent_bal_conc = pyro.sample(
                "latent_bal_conc",
                dist.LogNormal(bal_conc_loc, bal_conc_scale).to_event(1),
            )
            conc = kcat.new_ones(len(self.experiments), self.num_mics)
            conc[:, self.balanced_mics_idx] = latent_bal_conc
            conc[:, self.unbalanced_mics_idx] = unb_conc
            if self.has_fdx:
                fdx_ratio = pyro.sample(
                    "fdx_ratio", dist.LogNormal(fdx_contr, 0.1).to_event(1)
                )
                conc = torch.cat([conc, fdx_ratio], dim=1)

        free_enz_km_denom = get_free_enzyme_ratio_denom(
            conc,
            km,
            self.sub_conc_idx,
            self.sub_km_idx,
            self.prod_conc_idx,
            self.prod_km_idx,
            self.substrate_S,
            self.product_S,
        )
        free_enz_ki_denom = get_competitive_inhibition_denom(
            conc,
            ki,
            self.ki_conc_idx,
            self.ki_idx,
        ) if self.has_ci else 0
        free_enzyme_ratio = 1 / (free_enz_km_denom + free_enz_ki_denom)
        vmax = get_vmax(kcat, enz_conc)
        rev = get_reversibility(
            self.S_enz_thermo, dgr, conc, self.transported_charge, psi
        )
        sat = get_saturation(
            conc, km, free_enzyme_ratio, self.sub_conc_idx, self.sub_km_idx
        )
        allostery = (
            get_allostery(
                conc,
                free_enzyme_ratio,
                tc,
                dc,
                self.allostery_activation,
                self.allostery_idx,
                self.conc_allostery_idx,
                self.subunits,
            )
            if self.has_allostery
            else torch.ones_like(vmax)
        )
        flux = (
            vmax
            * rev
            * sat
            * allostery.reshape(len(self.experiments), len(self.sub_km_idx))
        )
        # TODO: small drain correction from config
        drain = (
            get_kinetic_multi_drain(
                kcat_drain,
                conc,
                self.sub_conc_drain_idx,
                self.prod_conc_drain_idx,
                self.substrate_drain_S,
                self.product_drain_S,
                1e-9,
            )
            if kcat_drain.size()[0]
            else None
        )
        all_flux = torch.cat([drain, flux], dim=1) if drain is not None else flux
        ssd = all_flux @ self.S.T[:, self.balanced_mics_idx]
        if penalize_ss:
            pyro.factor(
                "steady_state_dev",
                (ssd.abs() / (latent_bal_conc + 1e-13)).sum(),
                has_rsample=True,
            )

    def print_inputs(self):
        print(
            f"Exp: {self.experiments}\nNum reacs: {self.num_reactions}\n"
            f"Kcats: {self.kcat_loc};{self.kcat_scale}\n"
            f"Fluxes: {self.obs_fluxes};{self.obs_fluxes_std};{self.obs_fluxes_idx}"
        )

    def get_obs(self):
        return self.obs_fluxes, self.obs_conc
