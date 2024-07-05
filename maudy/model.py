from collections import defaultdict
from copy import deepcopy
from typing import Optional

import pandas as pd
import numpyro
import numpyro.distributions as dist
from numpyro.contrib.module import flax_module
from jax import numpy as jnp
from jax import lax
from maud.data_model.maud_input import MaudInput
from maud.data_model.experiment import MeasurementType
from .black_box import BaseConcCoder, BaseDecoder
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
numpyro.enable_x64(True)

def get_loc_from_mu_scale(mu: jnp.ndarray, scale: jnp.ndarray) -> jnp.ndarray:
    mu2 = jnp.pow(mu, 2)
    sigma_sq = (scale - 1) * mu2
    loc = jnp.log(jnp.pow(mu, 2) / jnp.sqrt(mu2 + jnp.pow(sigma_sq, 2)))
    return loc


def check_for_nans(data, name="Data"):
    assert jnp.all(jnp.isfinite(data)), f"{name} contains NaNs or infinities"


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


class Maudy:
    def __init__(self, maud_input: MaudInput, normalize: bool = False):
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
        """
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
        self.kcat_loc = jnp.array([
            kcats.loc[reac, "location"] for reac in enzymatic_reactions
        ])
        self.kcat_scale = jnp.array([
            kcats.loc[reac, "scale"] for reac in enzymatic_reactions
        ])
        # 2. enzyme concentrations
        ec = self.maud_params.conc_enzyme_train.prior
        enzyme_concs = pd.DataFrame(ec.location, index=ec.ids[0], columns=ec.ids[1])
        # sorted by reaction in the same order as kcats
        self.enzyme_concs_loc = jnp.array(enzyme_concs[enzymes].values)
        enzyme_concs = pd.DataFrame(ec.scale, index=ec.ids[0], columns=ec.ids[1])
        self.enzyme_concs_scale = jnp.array(enzyme_concs[enzymes].values)
        # 3. drains, fluxes are then concatenated in the model as drains + enzymatic_reations
        drain = self.maud_params.drain_train.prior
        drain_mean = pd.DataFrame(
            drain.location, index=drain.ids[0], columns=drain.ids[1]
        )
        self.drain_mean = jnp.array(drain_mean.values)
        self.drain_std = jnp.array(
            pd.DataFrame(drain.scale, index=drain.ids[0], columns=drain.ids[1]).values
        )
        # 4. dgfs
        dgf = self.maud_params.dgf.prior
        self.dgf_means = jnp.array(
            pd.Series(dgf.location, index=dgf.ids[0]).loc[mets]
        )
        dgf_cov = jnp.array(
            pd.DataFrame(dgf.covariance_matrix, index=dgf.ids[0], columns=dgf.ids[0])
            .loc[mets, mets]
            .values
        )
        self.dgf_cov = jnp.linalg.cholesky(dgf_cov)
        self.experiments = ec.ids[0]
        self.num_reactions = len(reactions)
        self.num_mics = len(mics)
        edge_ids = drain.ids[1] + [
            f"{e}_{r}" for e, r in zip(enzymes, enzymatic_reactions)
        ]
        unb_conc = self.maud_params.conc_unbalanced_train.prior
        self.balanced_mics_idx = jnp.array([
            i for i, met in enumerate(self.kinetic_model.mics) if met.balanced
        ], dtype=int)
        self.unbalanced_mics_idx = jnp.array([
            i for i, met in enumerate(self.kinetic_model.mics) if not met.balanced
        ], dtype=int)
        unb_mics = [mic for i, mic in enumerate(mics) if i in self.unbalanced_mics_idx]
        bal_mics = [mic for i, mic in enumerate(mics) if i in self.balanced_mics_idx]
        unb_config = maud_input._maudy_config.optimize_unbalanced_metabolites
        optimize_unbalanced = unb_config if unb_config is not None else []
        if optimize_unbalanced:
            opt_not_unb = set(optimize_unbalanced) - set(unb_mics)
            assert (
                not opt_not_unb
            ), f"{opt_not_unb} to be optimized are not unbalanced metabolites!"
        self.optimized_unbalanced_idx = jnp.array([
            i for i, met in enumerate(unb_mics) if met in optimize_unbalanced
        ])
        self.non_optimized_unbalanced_idx = jnp.array([
            i for i, met in enumerate(unb_mics) if met not in optimize_unbalanced
        ])
        self.unb_conc_loc = jnp.array(
            pd.DataFrame(
                unb_conc.location, index=unb_conc.ids[0], columns=unb_conc.ids[1]
            )
            .loc[self.experiments, unb_mics]
            .values
        )
        self.unb_conc_scale = jnp.array(
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
        self.bal_conc_mu = jnp.array([
            [
                conc_inits[(exp, mic)][0] if (exp, mic) in conc_inits else 1e-6
                for mic in bal_mics
            ]
            for exp in self.experiments
        ])
        self.bal_conc_scale = jnp.array([
            [
                conc_inits[(exp, mic)][1] if (exp, mic) in conc_inits else 1.0
                for mic in bal_mics
            ]
            for exp in self.experiments
        ])
        self.bal_conc_loc = get_loc_from_mu_scale(self.bal_conc_mu, self.bal_conc_scale)
        S = self.kinetic_model.stoichiometric_matrix.loc[mics, edge_ids]
        self.S = jnp.array(S.values)
        # S matrix only for stoichoimetric reactions (to calculate saturation and drG)
        self.S_enz = jnp.array(S.loc[:, ~S.columns.isin(drain.ids[1])].values)
        # S matrix used for thermo drG prime (reverisibility term), same as the one
        # for enzymatic reactions but it may be modified later if ferredoxin is present
        self.S_enz_thermo = jnp.array(
            S.loc[:, ~S.columns.isin(drain.ids[1])].values
        )
        mic_enz = S.loc[:, ~S.columns.isin(drain.ids[1])].index
        self.met_to_mic = jnp.array([
            mets.index(mic.split("_", 1)[0]) for mic in mic_enz
        ], dtype=int)
        water_and_trans = {
            reac.id: (reac.water_stoichiometry, reac.transported_charge)
            for reac in self.kinetic_model.reactions
        }
        self.water_stoichiometry = jnp.array([
            water_and_trans[r][0] for r in enzymatic_reactions
        ])
        self.transported_charge = jnp.array([
            water_and_trans[r][1] for r in enzymatic_reactions
        ])
        # 5. saturation, we need kms and indices
        reac_st = {reac.id: reac.stoichiometry for reac in self.kinetic_model.reactions}
        self.sub_conc_idx = [
            jnp.array([
                mics.index(met) for met, st in reac_st[reac].items() if st < 0
            ], dtype=int)
            for reac in enzymatic_reactions
        ]
        self.prod_conc_idx = [
            jnp.array([
                mics.index(met) for met, st in reac_st[reac].items() if st > 0
            ], dtype=int)
            for reac in enzymatic_reactions
        ]
        self.substrate_S = [
            jnp.array([-st for _, st in reac_st[reac].items() if st < 0], dtype=int)
            for reac in enzymatic_reactions
        ]
        self.product_S = [
            jnp.array([st for _, st in reac_st[reac].items() if st > 0], dtype=int)
            for reac in enzymatic_reactions
        ]
        # the same but for drains
        self.sub_conc_drain_idx = [
            jnp.array([
                mics.index(met) for met, st in reac_st[reac].items() if st < 0
            ], dtype=int)
            for reac in drain.ids[1]
        ]
        self.prod_conc_drain_idx = [
            jnp.array([
                mics.index(met) for met, st in reac_st[reac].items() if st > 0
            ], dtype=int)
            for reac in drain.ids[1]
        ]
        self.substrate_drain_S = [
            jnp.array([-st for _, st in reac_st[reac].items() if st < 0], dtype=int)
            for reac in drain.ids[1]
        ]
        self.product_drain_S = [
            jnp.array([st for _, st in reac_st[reac].items() if st > 0], dtype=int)
            for reac in drain.ids[1]
        ]
        # the kms
        kms = self.maud_params.km.prior
        km_map = {}
        for i, km_id in enumerate(kms.ids[0]):
            enzyme, mic = km_id.split("_", 1)
            km_map[(enzyme, mic)] = i
        self.km_loc = jnp.array(kms.location)
        self.km_scale = jnp.array(kms.scale)
        self.sub_km_idx = [
            jnp.array([
                km_map[(enz, met)] for met, st in reac_st[reac].items() if st < 0
            ], dtype=int)
            for enz, reac in zip(enzymes, enzymatic_reactions)
        ]
        # if the enz, mic is not in km_map, it is a irreversible reaction
        self.prod_km_idx = [
            jnp.array([
                km_map[(enz, met)]
                for met, st in reac_st[reac].items()
                if st > 0 and (enz, met) in km_map
            ], dtype=int)
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
            jnp.unique(jnp.array(all_sub_idx))
        ), "The indexing on the Km values went wrong for the substrates"
        assert len(all_prod_idx) == len(
            jnp.unique(jnp.array(all_prod_idx))
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
            self.ki_loc = jnp.array(kis.location)
            self.ki_scale = jnp.array(kis.scale)
            self.ki_idx = [
                jnp.array([i_ki for i_ki, _ in ki_map[enz]], dtype=int) for enz in enzymes
            ]
            self.ki_conc_idx = [
                jnp.array(
                    [i_conc for _, i_conc in ki_map[enz]] if enz in ki_map else []
                , dtype=int)
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
            self.dc_loc = jnp.array(dc.location)
            self.dc_scale = jnp.array(dc.scale)
            self.tc_loc = jnp.array(tc.location)
            self.tc_scale = jnp.array(tc.scale)
            self.allostery_idx = jnp.array([
                dc_map[enz][0] for enz in enzymes if enz in dc_map
            ], dtype=int)
            self.conc_allostery_idx = jnp.array([
                dc_map[enz][1] for enz in enzymes if enz in dc_map
            ], dtype=int)
            self.allostery_activation = jnp.array([
                dc_map[enz][2] == "activation" for enz in enzymes if enz in dc_map
            ], dtype=int)
            self.subunits = jnp.array([
                next(
                    kin_enz.subunits
                    for kin_enz in self.kinetic_model.enzymes
                    if kin_enz.id == enz
                )
                for enz in enzymes
                if enz in dc_map
            ], dtype=int)

        self.obs_fluxes = jnp.array([
            [
                meas.value
                for meas in exp.measurements
                if meas.target_type == MeasurementType.FLUX
            ]
            for exp in self.maud_params.experiments
        ])
        self.obs_fluxes_std = jnp.array([
            [
                meas.error_scale
                for meas in exp.measurements
                if meas.target_type == MeasurementType.FLUX
            ]
            for exp in self.maud_params.experiments
        ])
        self.obs_conc = jnp.array([
            [
                conc_obs[(e, mic)][0] if (e, mic) in conc_obs else float("nan")
                for mic in mics
            ]
            for e in self.experiments
        ])
        self.obs_conc_std = jnp.array([
            [
                conc_obs[(e, mic)][1] if (e, mic) in conc_obs else float("nan")
                for mic in mics
            ]
            for e in self.experiments
        ])
        idx = jnp.array([
            [
                enzymatic_reactions.index(meas.reaction)
                for meas in exp.measurements
                if meas.target_type == MeasurementType.FLUX
            ]
            for exp in self.maud_params.experiments
        ], dtype=int)
        self.num_obs_fluxes = len(idx[0])
        self.obs_fluxes_idx = (
            [i for i, exp in enumerate(idx) for _ in exp],
            [i for exp in idx for i in exp],
        )
        self.obs_conc_mask = ~jnp.isnan(self.obs_conc_std)
        # Special case of ferredoxin: we want to add a per-experiment
        # concentration ratio parameter (output of NN) and the dGf difference
        self.fdx_stoichiometry = jnp.zeros_like(self.water_stoichiometry)
        fdx = maud_input._maudy_config.ferredoxin
        if fdx is not None:
            # first check if all reactions have a correct identifier to catch user typos
            fdx_not_reac = set(fdx.keys()) - set(enzymatic_reactions)
            assert (
                len(fdx_not_reac) == 0
            ), f"{fdx_not_reac} with ferredoxin not in {enzymatic_reactions}"
            self.fdx_stoichiometry = jnp.array([
                fdx[r] if r in fdx else 0 for r in enzymatic_reactions
            ])
            # add row for S matrix to calculate DrG prime
            self.S_enz_thermo = jnp.concat(
                [self.S_enz_thermo, jnp.expand_dims(self.fdx_stoichiometry, axis=0)], axis=0
            )
        self.enzymatic_reactions = enzymatic_reactions
        self.has_fdx = any(st != 0 for st in self.fdx_stoichiometry)
        nn_config = maud_input._maudy_config.neural_network
        # Setup the various neural networks
        # when there are very small values, we need to normalize the conc so that
        # it does not explode
        all_concs = jnp.concat((jnp.log(self.obs_conc[self.obs_conc_mask]), self.unb_conc_loc.flatten()))
        min_max = (all_concs.min().item() - 0.6, all_concs.max().item() + 0.6) if normalize else None
        self.init_latent = all_concs.mean().item()
        self.normalize = normalize
        self.decoder_args = dict(
            met_dim=len(self.balanced_mics_idx),
            unb_dim=self.unb_conc_loc.shape[1],
            enz_dim=len(enzymatic_reactions),
            drain_dim=self.drain_mean.shape[1],
            normalize=min_max,
        )
        self.encoder_args = dict(
            met_dims=[len(self.non_optimized_unbalanced_idx)]
            + nn_config.met_dims
            + [len(self.balanced_mics_idx)],
            reac_dim=len(enzymatic_reactions),
            km_dims=[len(self.km_loc)]
            + nn_config.km_dims
            + [len(self.balanced_mics_idx)],
            drain_dim=self.drain_mean.shape[1] if self.drain_mean.size != 0 else 0,
            ki_dim=self.ki_loc.shape[0] if hasattr(self, "ki_loc") else 0,
            tc_dim=self.tc_loc.shape[0] if hasattr(self, "tc_loc") else 0,
            # batch norm and dropout won't work without a batch dim
            drop_out=len(self.experiments) > 1,
            batchnorm=len(self.experiments) > 1,
            normalize=min_max,
        )
        # if self.has_fdx:
        #     fdx_head(nn_encoder)
        # self.has_opt_unb = self.optimized_unbalanced_idx.size != 0
        # if self.has_opt_unb:
        #     unb_opt_head(nn_encoder, unb_dim=self.optimized_unbalanced_idx.shape[-1])
        # self.concoder = nn_encoder

    def model(
        self,
        obs_flux: Optional[jnp.ndarray] = None,
        obs_conc: Optional[jnp.ndarray] = None,
        penalize_ss: bool = True,
        annealing_factor: float = 1.0,
    ):
        """Describe the generative model."""
        # Register various nn.Modules (neural networks) with Pyro

        # experiment-indepedent variables
        kcat = numpyro.sample(
            "kcat", dist.LogNormal(self.kcat_loc, self.kcat_scale).to_event(1)
        )
        dgf = numpyro.sample(
            "dgf", dist.MultivariateNormal(self.dgf_means, scale_tril=self.dgf_cov)
        )
        dgf = dgf.reshape(-1)
        fdx_contr = (
            numpyro.sample("fdx_contr", dist.Normal(77, 1))
            if self.has_fdx
            else jnp.array([0.0])
        )
        dgr = numpyro.deterministic(
            "dgr",
            get_dgr(
                self.S_enz,
                dgf[self.met_to_mic],
                self.water_stoichiometry,
                self.fdx_stoichiometry,
                fdx_contr,
            ),
        )
        km = numpyro.sample("km", dist.LogNormal(self.km_loc, self.km_scale).to_event(1))
        rest = jnp.array([])
        if self.has_ci:
            ki = numpyro.sample(
                "ki", dist.LogNormal(self.ki_loc, self.ki_scale).to_event(1)
            )
            rest = ki
        if self.has_allostery:
            dc = numpyro.sample(
                "dc", dist.LogNormal(self.dc_loc, self.dc_scale).to_event(1)
            )
            tc = numpyro.sample(
                "tc", dist.LogNormal(self.tc_loc, self.tc_scale).to_event(1)
            )
            rest = jnp.concat([rest, tc, dc], axis=-1)
        # TODO: need to take this from the config (and done in th epalte)
        psi = numpyro.sample(
            "psi", dist.Normal(-0.110, 0.01)
        )
        with numpyro.plate("experiment", size=len(self.experiments)):
            enzyme_conc = numpyro.sample(
                "enzyme_conc",
                dist.LogNormal(self.enzyme_concs_loc, self.enzyme_concs_scale).to_event(
                    1
                ),
            )
            kcat_drain = (
                numpyro.sample(
                    "kcat_drain",
                    dist.Normal(self.drain_mean, self.drain_std).to_event(1),
                )
                if self.drain_mean.shape[1]
                else jnp.array([])
            )
            unb_conc = numpyro.sample(
                "unb_conc",
                dist.LogNormal(
                    self.unb_conc_loc,
                    self.unb_conc_scale,
                ).to_event(1),
            )
            latent_bal_conc = numpyro.sample(
                "latent_bal_conc",
                dist.LogNormal(jnp.full_like(self.obs_conc[:, self.balanced_mics_idx], self.init_latent), 1.0).to_event(
                    1
                ),
            )
            encoder = flax_module(
                "encoder",
                BaseDecoder(**self.decoder_args),
                jnp.zeros_like(latent_bal_conc), jnp.ones_like(unb_conc), jnp.ones_like(enzyme_conc), jnp.ones_like(kcat_drain)
            )
            ln_bal_conc = numpyro.deterministic("ln_bal_conc", encoder(latent_bal_conc, unb_conc, enzyme_conc, kcat_drain))
            conc = jnp.ones((len(self.experiments), self.num_mics))
            conc = conc.at[:, self.balanced_mics_idx].set(jnp.exp(ln_bal_conc))
            conc = conc.at[:, self.unbalanced_mics_idx].set(unb_conc)
            conc_comp = conc
            # if self.has_fdx:
            #     fdx_ratio = numpyro.sample(
            #         "fdx_ratio",
            #         dist.LogNormal(0.0, 0.1).to_event(1),
            #     )
            #     conc = jnp.concat([conc, fdx_ratio], axis=1)
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
                numpyro.deterministic(
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
            free_enzyme_ratio = numpyro.deterministic(
                "free_enzyme_ratio",
                1 / (free_enz_km_denom + free_enz_ki_denom),
            )
            vmax = numpyro.deterministic("vmax", get_vmax(kcat, enzyme_conc))
            rev = numpyro.deterministic(
                "rev",
                get_reversibility(
                    self.S_enz_thermo, dgr, conc, self.transported_charge, psi
                ),
            )
            sat = numpyro.deterministic(
                "sat",
                get_saturation(
                    conc, km, free_enzyme_ratio, self.sub_conc_idx, self.sub_km_idx
                ),
            )
            allostery = (
                numpyro.deterministic(
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
                else jnp.ones_like(vmax)
            )
            flux = numpyro.deterministic(
                "flux", vmax * rev * sat * allostery
            ).reshape(len(self.experiments), len(self.sub_km_idx))
            true_obs_flux = flux[self.obs_fluxes_idx]
            # Ensure true_obs_flux has shape [num_experiments, num_obs_fluxes]
            true_obs_flux = true_obs_flux.reshape(len(self.experiments), -1)
            # if true_obs_flux.ndim == 1:
            #     true_obs_flux = true_obs_flux.unsqueeze(-1)

            # TODO: small drain correction from config
            drain = (
                numpyro.deterministic(
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
                if kcat_drain.size != 0
                else None
            )
            all_flux = jnp.concat([drain, flux], axis=1) if drain is not None else flux
            ssd = numpyro.deterministic(
                "ssd", all_flux @ self.S.T[:, self.balanced_mics_idx]
            )
            numpyro.sample(
                "y_flux_train",
                dist.Normal(true_obs_flux, self.obs_fluxes_std * annealing_factor).to_event(1),
                obs=obs_flux,
            )
            numpyro.sample(
                "y_conc_train",
                dist.LogNormal(
                    jnp.log(conc_comp)[self.obs_conc_mask],
                    self.obs_conc_std[self.obs_conc_mask] / annealing_factor,
                ).to_event(1),
                obs=obs_conc[self.obs_conc_mask] if obs_conc is not None else None,
            )
            ssd_factor = numpyro.deterministic(
                "ssd_factor",
                0.5 * (jnp.log(jnp.abs(ssd) + 1e-14) - ln_bal_conc).clip(-6.90775, 6.90775).sum(axis=-1),
            )
            ssd_factor = lax.select(penalize_ss, ssd_factor, jnp.zeros_like(ssd_factor))
            numpyro.factor(
                "steady_state_dev",
                 -ssd_factor,
            )

    # The guide specifies the variational distribution
    def guide(
        self,
        obs_flux: Optional[jnp.ndarray] = None,
        obs_conc: Optional[jnp.ndarray] = None,
        penalize_ss: bool = False,
        annealing_factor: float = 1.0,
    ):
        """Establish the variational distributions for SVI."""
        dgf_param_loc = numpyro.param("dgf_loc", self.dgf_means)
        dgf = numpyro.sample(
            "dgf", dist.MultivariateNormal(dgf_param_loc, scale_tril=self.dgf_cov)
        )
        fdx_contr_loc = numpyro.param("fdx_contr", jnp.array([77.0]))
        fdx_contr_scale = numpyro.param(
            "fdx_contr_scale", jnp.array([1.0]), constraint=Positive
        )
        # Perform both sampling operations
        fdx_contr_sampled = numpyro.sample("fdx_contr_sampled", dist.Normal(fdx_contr_loc, fdx_contr_scale))
        fdx_contr_default = jnp.array([0.0])

        # Select the correct value based on the condition
        fdx_contr = lax.select(self.has_fdx, fdx_contr_sampled, fdx_contr_default)
        kcat_param_loc = numpyro.param("kcat_loc", self.kcat_loc)
        kcat = numpyro.sample(
            "kcat", dist.LogNormal(kcat_param_loc, self.kcat_scale).to_event(1)
        )
        km_loc = numpyro.param("km_loc", self.km_loc)
        km_scale = numpyro.param("km_scale", self.km_scale, constraint=Positive)
        km = numpyro.sample("km", dist.LogNormal(km_loc, km_scale).to_event(1))
        dgr = get_dgr(
            self.S_enz,
            dgf[self.met_to_mic],
            self.water_stoichiometry,
            self.fdx_stoichiometry,
            fdx_contr,
        )
        rest = jnp.array([])
        if self.has_ci:
            ki_loc = numpyro.param("ki_loc", self.ki_loc)
            ki_scale = numpyro.param("ki_scale", self.ki_scale, constraint=Positive)
            ki = numpyro.sample("ki", dist.LogNormal(ki_loc, ki_scale).to_event(1))
            rest = ki
        if self.has_allostery:
            dc_loc = numpyro.param("dc_loc", self.dc_loc)
            dc_scale = numpyro.param("dc_scale", self.dc_scale, constraint=Positive)
            tc_loc = numpyro.param("tc_loc", self.tc_loc)
            tc_scale = numpyro.param("tc_scale", self.tc_scale, constraint=Positive)
            dc = numpyro.sample("dc", dist.LogNormal(dc_loc, dc_scale).to_event(1))
            tc = numpyro.sample("tc", dist.LogNormal(tc_loc, tc_scale).to_event(1))
            rest = jnp.concat([rest, tc, dc])

        psi_mean = numpyro.param("psi_mean", jnp.array(-0.110))
        numpyro.sample(
            "psi", dist.Normal(psi_mean, jnp.array(0.01))
        )
        with numpyro.plate("experiment", size=len(self.experiments)):
            enzyme_concs_param_loc = numpyro.param(
                "enzyme_concs_loc", self.enzyme_concs_loc, event_dim=1
            )
            enz_conc = numpyro.sample(
                "enzyme_conc",
                dist.LogNormal(
                    enzyme_concs_param_loc, self.enzyme_concs_scale
                ).to_event(1),
            )
            drain_mean = numpyro.param("drain_mean", self.drain_mean, event_dim=1)
            drain_std = numpyro.param(
                "drain_std", self.drain_std, constraint=Positive, event_dim=1
            )
            kcat_drain = (
                numpyro.sample(
                    "kcat_drain",
                    dist.Normal(drain_mean, drain_std).to_event(1),
                )
                if self.drain_mean.shape[1]
                else jnp.array([])
            )
            unb_conc_param_loc = numpyro.param(
                "unb_conc_param_loc",
                self.unb_conc_loc,
                event_dim=1,
            )
            concoder_net = flax_module(
                "concoder",
                BaseConcCoder(**self.encoder_args),
                jnp.zeros_like(unb_conc_param_loc), jnp.zeros_like(dgr), jnp.ones_like(enz_conc), jnp.ones_like(kcat), jnp.ones_like(kcat_drain), jnp.ones_like(km), jnp.ones_like(km),
                apply_rng=["dropout"],
            )
            # in reverse order that additional head outputs may have been added
            # if self.has_opt_unb:
            #     unb_optimized = concoder_output.pop()
            #     unb_conc_param_loc_full[:, self.optimized_unbalanced_idx] = (
            #         unb_optimized
            #     )
            # if self.has_fdx:
            #     fdx_ratio = concoder_output.pop()
            latent_bal_conc_loc, bal_conc_scale = concoder_net(unb_conc_param_loc, dgr, enz_conc, kcat, kcat_drain, km, rest, rngs={"dropout": numpyro.prng_key()})
            numpyro.sample(
                "unb_conc",
                dist.LogNormal(
                    unb_conc_param_loc, self.unb_conc_scale
                ).to_event(1),
            )
            numpyro.sample(
                "latent_bal_conc",
                dist.LogNormal(latent_bal_conc_loc, bal_conc_scale).to_event(1),
            )
            # if self.has_fdx:
            #     fdx_ratio = numpyro.sample(
            #         "fdx_ratio", dist.LogNormal(fdx_ratio, 0.1).to_event(1)
            #     )

    def print_inputs(self):
        print(
            f"Exp: {self.experiments}\nNum reacs: {self.num_reactions}\n"
            f"Kcats: {self.kcat_loc};{self.kcat_scale}\n"
            f"Fluxes: {self.obs_fluxes};{self.obs_fluxes_std};{self.obs_fluxes_idx}"
        )

    def get_obs(self):
        return self.obs_fluxes, self.obs_conc
