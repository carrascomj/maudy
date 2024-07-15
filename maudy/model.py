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
from .black_box import BaseConcCoder, BaseDecoder, Norm, fdx_head, unb_opt_head
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


def pretty_print_tensor_with_big_brackets(tensor):
    # Ensure the tensor is a float type for proper formatting
    tensor = tensor.float()
    # Convert tensor to a list of lists for easy printing
    tensor_list = tensor.tolist()
    # Pretty print each element with 4 decimal places
    formatted_tensor = [[f"{element:.4f}" for element in row] for row in tensor_list]

    # Define the big bracket components
    top_bracket = "⎡ "
    middle_bracket = "⎢ "
    bottom_bracket = "⎣ "

    # Print the formatted tensor with big brackets
    for i, row in enumerate(formatted_tensor):
        if i == 0:
            print(top_bracket + "  ".join(row) + " ⎤")
        elif i == len(formatted_tensor) - 1:
            print(bottom_bracket + "  ".join(row) + " ⎦")
        else:
            print(middle_bracket + "  ".join(row) + " ⎥")


class Maudy(nn.Module):
    def __init__(self, maud_input: MaudInput, normalize: bool = False, quench: bool = False):
        """Initialize the priors of the model.

        maud_input: MaudInput
            a `MaudInput` with an optionally injected `._fdx_stoichiometry`
            dict atttribute with identifiers as keys and its ferredoxin
            stoichiometry as values (defaults 0 for reactions not in the dict).
        optimize_unbalanced: Optional[list[str]]
            unbalanced metabolite-in-compartment identifiers to infer as the
            output of the neural network.
        normalize: bool, default=False
            whether to normalize the input and output of the neural network.
            The input is normalize such that positive values are turned to log-space
            and drains and fluxes are multiplied by 1e6 (mumol). The output is
            clamped between 0.6 orders of magnitude of the higher and lowest
            observed or prior concentrations.
        quench: bool, default=False
            whether to add a model that learns a transformation from steady
            state concentrations - that fits the SSD and fluxes - to observed
            concentrations.
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
        unb_config = maud_input._maudy_config.optimize_unbalanced_metabolites
        optimize_unbalanced = unb_config if unb_config is not None else []
        if optimize_unbalanced:
            opt_not_unb = set(optimize_unbalanced) - set(unb_mics)
            assert (
                not opt_not_unb
            ), f"{opt_not_unb} to be optimized are not unbalanced metabolites!"
        self.optimized_unbalanced_idx = torch.LongTensor([
            i for i, met in enumerate(unb_mics) if met in optimize_unbalanced
        ])
        self.non_optimized_unbalanced_idx = torch.LongTensor([
            i for i, met in enumerate(unb_mics) if met not in optimize_unbalanced
        ])
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
        self.bal_conc_mu = torch.FloatTensor([
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
        # the same but for drains
        self.sub_conc_drain_idx = [
            torch.LongTensor([
                mics.index(met) for met, st in reac_st[reac].items() if st < 0
            ])
            for reac in drain.ids[1]
        ]
        self.prod_conc_drain_idx = [
            torch.LongTensor([
                mics.index(met) for met, st in reac_st[reac].items() if st > 0
            ])
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
            self.allostery_idx = torch.LongTensor([
                dc_map[enz][0] for enz in enzymes if enz in dc_map
            ])
            self.conc_allostery_idx = torch.LongTensor([
                dc_map[enz][1] for enz in enzymes if enz in dc_map
            ])
            self.allostery_activation = torch.BoolTensor([
                dc_map[enz][2] == "activation" for enz in enzymes if enz in dc_map
            ])
            self.subunits = torch.IntTensor([
                next(
                    kin_enz.subunits
                    for kin_enz in self.kinetic_model.enzymes
                    if kin_enz.id == enz
                )
                for enz in enzymes
                if enz in dc_map
            ])

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
            [
                conc_obs[(e, mic)][0] if (e, mic) in conc_obs else float("nan")
                for mic in mics
            ]
            for e in self.experiments
        ])
        self.obs_conc_std = torch.FloatTensor([
            [
                conc_obs[(e, mic)][1] if (e, mic) in conc_obs else float("nan")
                for mic in mics
            ]
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
        self.obs_conc_mask = ~torch.isnan(self.obs_conc_std)
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
            self.fdx_stoichiometry = torch.FloatTensor([
                fdx[r] if r in fdx else 0 for r in enzymatic_reactions
            ])
            # add row for S matrix to calculate DrG prime
            self.S_enz_thermo = torch.cat(
                [self.S_enz_thermo, self.fdx_stoichiometry.unsqueeze(0)], dim=0
            )
        self.enzymatic_reactions = enzymatic_reactions
        self.has_fdx = any(st != 0 for st in self.fdx_stoichiometry)
        nn_config = maud_input._maudy_config.neural_network
        # Setup the various neural networks
        # when there are very small values, we need to normalize the conc so that
        # it does not explode
        all_concs = torch.cat((self.obs_conc[self.obs_conc_mask].log(), self.unb_conc_loc.flatten()))
        min_max = (all_concs.min().item() - 3, all_concs.max().item() + 2) if normalize else None
        self.init_latent = all_concs.mean().item()
        self.normalize = normalize
        self.decoder = BaseDecoder(
            met_dim=len(self.balanced_mics_idx),
            unb_dim=self.unb_conc_loc.shape[1],
            enz_dim=len(enzymatic_reactions),
            drain_dim=self.drain_mean.shape[1],
            normalize=min_max,
            batchnorm=len(self.experiments) > 1,
        )
        nn_encoder = BaseConcCoder(
            met_dims=[len(self.non_optimized_unbalanced_idx)]
            + nn_config.met_dims
            + [len(self.balanced_mics_idx)],
            reac_dim=len(enzymatic_reactions),
            km_dims=[len(self.km_loc)]
            + nn_config.km_dims
            + [len(self.balanced_mics_idx)],
            drain_dim=self.drain_mean.shape[1] if len(self.drain_mean.size()) else 0,
            ki_dim=self.ki_loc.shape[0] if hasattr(self, "ki_loc") else 0,
            tc_dim=self.tc_loc.shape[0] if hasattr(self, "tc_loc") else 0,
            # batch norm and dropout won't work without a batch dim
            drop_out=len(self.experiments) > 1,
            batchnorm=len(self.experiments) > 1,
            normalize=min_max,
        )
        if self.has_fdx:
            fdx_head(nn_encoder)
        self.has_opt_unb = self.optimized_unbalanced_idx.numel() != 0
        if self.has_opt_unb:
            unb_opt_head(nn_encoder, unb_dim=self.optimized_unbalanced_idx.shape[-1])
        self.concoder = nn_encoder
        met_dim = len(self.balanced_mics_idx)
        self.quench = (
        (
            lambda _: torch.zeros(
                (len(self.experiments), met_dim), device=self.water_stoichiometry.device
            )
        )
        if not quench
        else nn.Sequential(
            *[
                nn.Sequential(nn.Linear(in_dim, out_dim), Norm(), nn.ReLU())
                for in_dim, out_dim in zip(
                    [met_dim] + nn_config.quench_dims, nn_config.quench_dims + [met_dim]
                )
            ]
        )
        )

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
        penalize_ss: bool = False,
        annealing_factor: float = 1.0,
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
        # TODO: need to take this from the config (and done in th epalte)
        psi = pyro.sample(
            "psi", dist.Normal(self.float_tensor(-0.110), self.float_tensor(0.01))
        )
        with pyro.plate("experiment", size=len(self.experiments)) as idx:
            enzyme_conc = pyro.sample(
                "enzyme_conc",
                dist.LogNormal(self.enzyme_concs_loc, self.enzyme_concs_scale).to_event(
                    1
                ),
            )
            kcat_drain = (
                pyro.sample(
                    "kcat_drain",
                    dist.Normal(self.drain_mean, self.drain_std).to_event(1),
                )
                if self.drain_mean.shape[1]
                else self.float_tensor([])
            )
            unb_conc = pyro.sample(
                "unb_conc",
                dist.LogNormal(
                    self.unb_conc_loc,
                    self.unb_conc_scale,
                ).to_event(1),
            )
            with pyro.poutine.scale(scale=annealing_factor):
                latent_bal_conc = pyro.sample(
                    "latent_bal_conc",
                    dist.LogNormal(torch.full_like(self.obs_conc[:, self.balanced_mics_idx], self.init_latent), 1.0).to_event(
                        1
                    ),
                )
            ln_bal_conc = pyro.deterministic(
                "ln_bal_conc",
                self.decoder(latent_bal_conc, unb_conc, enzyme_conc, kcat_drain),
                event_dim=1,
            )
            conc = kcat.new_ones(len(self.experiments), self.num_mics)
            conc[:, self.balanced_mics_idx] = ln_bal_conc.exp()
            conc[:, self.unbalanced_mics_idx] = unb_conc
            if self.has_fdx:
                fdx_ratio = pyro.sample(
                    "fdx_ratio",
                    dist.LogNormal(self.float_tensor([0.0]), 0.1).to_event(1),
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
                    event_dim=1,
                )
                if self.has_ci
                else 0
            )
            free_enzyme_ratio = pyro.deterministic(
                "free_enzyme_ratio",
                1 / (free_enz_km_denom + free_enz_ki_denom),
                event_dim=1,
            )
            vmax = pyro.deterministic("vmax", get_vmax(kcat, enzyme_conc), event_dim=1)
            rev = pyro.deterministic(
                "rev",
                get_reversibility(
                    self.S_enz_thermo, dgr, conc, self.transported_charge, psi
                ),
                event_dim=1,
            )
            sat = pyro.deterministic(
                "sat",
                get_saturation(
                    conc, km, free_enzyme_ratio, self.sub_conc_idx, self.sub_km_idx
                ),
                event_dim=1,
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
                    event_dim=1,
                )
                if self.has_allostery
                else torch.ones_like(vmax)
            )
            flux = pyro.deterministic(
                "flux", vmax * rev * sat * allostery, event_dim=1
            ).reshape(len(self.experiments), len(self.sub_km_idx))
            true_obs_flux = flux[self.obs_fluxes_idx]
            # Ensure true_obs_flux has shape [num_experiments, num_obs_fluxes]
            true_obs_flux = true_obs_flux.reshape(len(self.experiments), -1)
            if true_obs_flux.ndim == 1:
                true_obs_flux = true_obs_flux.unsqueeze(-1)

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
                    event_dim=1,
                )
                if kcat_drain.size()[0]
                else None
            )
            all_flux = torch.cat([drain, flux], dim=1) if drain is not None else flux
            ssd = pyro.deterministic(
                "ssd", all_flux @ self.S.T[:, self.balanced_mics_idx], event_dim=1
            )
            pyro.sample(
                "y_flux_train",
                dist.Normal(true_obs_flux, self.obs_fluxes_std * annealing_factor).to_event(1),
                obs=obs_flux,
            )
            # quenched concentrations
            conc_comp = kcat.new_ones(len(self.experiments), self.num_mics)
            quench_correction = pyro.deterministic("quench_correction", self.quench(ln_bal_conc))
            conc_comp[:, self.balanced_mics_idx] = (ln_bal_conc + quench_correction).exp()
            conc_comp[:, self.unbalanced_mics_idx] = unb_conc
            if penalize_ss:
                ssd_factor = pyro.deterministic(
                    "ssd_factor",
                    ssd.abs() / (ln_bal_conc.exp() + 1e-13),
                    event_dim=1,
                )
                pyro.factor(
                    "steady_state_dev",
                    -ssd_factor.clamp(1e-3, 1000).sum(dim=-1),
                )
            # each experiment, once selected by the mask, might have different
            # dimensions, thus the loop
            for i in idx:
                pyro.sample(
                    f"y_conc_train_{i}",
                    dist.LogNormal(
                        conc_comp[i].log()[self.obs_conc_mask[i]],
                        self.obs_conc_std[i][self.obs_conc_mask[i]] / annealing_factor,
                    ).to_event(1),
                    obs=obs_conc[i][self.obs_conc_mask[i]] if obs_conc is not None else None,
                )
            # if penalize_ss:
            #     ssd_factor = pyro.deterministic(
            #         "ssd_factor",
            #         # 0.5 * torch.exp(2 * (torch.log(ssd.abs() + 1e-14) - ln_bal_conc).clamp(-6.90775, 6.90775)).sum(dim=-1),
            #         # torch.log(ssd.abs() + 1e-12).clamp(ln_bal_conc - 6.907755, None).sum(dim=-1),
            #         1000 * ssd.abs().clamp(1e-11, None).sum(dim=-1),
            #         # 1000 * (torch.log(ssd.abs() + 1e-14) - ln_bal_conc).clamp(-6.907755, None),
            #         event_dim=1,
            #     )
            #     pyro.factor(
            #         "steady_state_dev",
            #          -ssd_factor.sum(dim=-1),
            #     )

    def float_tensor(self, x) -> torch.Tensor:
        return torch.tensor(x, device=self.water_stoichiometry.device)

    # The guide specifies the variational distribution
    def guide(
        self,
        obs_flux: Optional[torch.FloatTensor] = None,
        obs_conc: Optional[torch.FloatTensor] = None,
        penalize_ss: bool = False,
        annealing_factor: float = 1.0,
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

        psi_mean = pyro.param("psi_mean", self.float_tensor(-0.110))
        pyro.sample(
            "psi", dist.Normal(psi_mean, self.float_tensor(0.01))
        )
        with pyro.plate("experiment", size=len(self.experiments)):
            enzyme_concs_param_loc = pyro.param(
                "enzyme_concs_loc", self.enzyme_concs_loc, event_dim=1
            )
            enz_conc = pyro.sample(
                "enzyme_conc",
                dist.LogNormal(
                    enzyme_concs_param_loc, self.enzyme_concs_scale
                ).to_event(1),
            )
            drain_mean = pyro.param("drain_mean", lambda: self.drain_mean, event_dim=1)
            drain_std = pyro.param(
                "drain_std", lambda: self.drain_std, constraint=Positive, event_dim=1
            )
            kcat_drain = (
                pyro.sample(
                    "kcat_drain",
                    dist.Normal(drain_mean, drain_std).to_event(1),
                )
                if self.drain_mean.shape[1]
                else self.float_tensor([])
            )
            unb_conc_param_loc = pyro.param(
                "unb_conc_param_loc",
                self.unb_conc_loc[:, self.non_optimized_unbalanced_idx],
                event_dim=1,
            )
            concoder_output = self.concoder(
                unb_conc_param_loc, dgr, enz_conc, kcat, kcat_drain, km, rest
            )
            unb_conc_param_loc_full = torch.full_like(self.unb_conc_loc, 1.0)
            unb_conc_param_loc_full[:, self.non_optimized_unbalanced_idx] = (
                unb_conc_param_loc
            )
            # in reverse order that additional head outputs may have been added
            if self.has_opt_unb:
                unb_optimized = concoder_output.pop()
                unb_conc_param_loc_full[:, self.optimized_unbalanced_idx] = (
                    unb_optimized
                )
            if self.has_fdx:
                fdx_ratio = concoder_output.pop()
            latent_bal_conc_loc, bal_conc_scale = concoder_output
            pyro.sample(
                "unb_conc",
                dist.LogNormal(
                    unb_conc_param_loc_full, self.unb_conc_scale
                ).to_event(1),
            )
            with pyro.poutine.scale(scale=annealing_factor):
                pyro.sample(
                    "latent_bal_conc",
                    dist.LogNormal(latent_bal_conc_loc, bal_conc_scale).to_event(1),
                )
            if self.has_fdx:
                fdx_ratio = pyro.sample(
                    "fdx_ratio", dist.LogNormal(fdx_ratio, 0.1).to_event(1)
                )

    def print_inputs(self):
        print(
            f"Exp: {self.experiments}\nNum reacs: {self.num_reactions}\n"
            f"Kcats: {self.kcat_loc};{self.kcat_scale}\n"
            f"Fluxes: {self.obs_fluxes};{self.obs_fluxes_std};{self.obs_fluxes_idx}"
        )

    def get_obs(self):
        return self.obs_fluxes, self.obs_conc
