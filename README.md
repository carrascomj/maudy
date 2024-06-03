# maudy

We have a kinetics-informed Bayesian neural network that solves accepts three kinds of variables:

1. Boundary variables.
2. Observed variables.
3. Innacessible variables.

First, the model samples the priors (both boudaries and observed true values).
Second, a neural network samples the innacessible variables. Third, the fluxes
are computed given all variables. Finally, the likelihood is computed for the
observed variables and, for the observed and innacessible variables that refer
to concentrations, a loss on the steady state $Sv[~Boundary] ~ 0$ is applied.


## Roadmap

* [x] Model that implements a vmax-based flux model with priors only for $k_{cat}$ values and enzyme concentrations. The likelihood is computed only for the fluxes. The neural network outputs a flux correction.
* [x] Model that implements a vmax-based flux model with priors only for $k_{cat}$ values and enzyme concentrations. The likelihood is computed for fluxes and steady state. The neural network outputs a flux correction.
* [ ] Model that implements all elements from the modular rate law in Maud but the regulatory terms. The likelihood is computed for fluxes, concentrations and steady state. The neural network outputs innacesible variables only referring to concentrations.
* [ ] The same but with the membrane potential, the special case of ferredoxin only affecting dG and the ratio of ferredoxin being an innacessible variable.
