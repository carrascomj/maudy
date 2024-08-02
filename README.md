# maudy

We have a kinetics-informed Bayesian neural network that solves accepts three kinds of variables:

1. Boundary variables.
2. Observed variables.
3. Innacessible variables.

First, the model samples the priors (boudaries of ODE).
Second, a neural network samples the innacessible variables and balanced concentrations (substituing the ODE). Third, the fluxes
are computed given all variables. Finally, the likelihood is computed for the
observed variables and, for the observed and innacessible variables that refer
to concentrations, a loss on the steady state $Sv[~Boundary] ~ 0$ is applied.

## Installation

```bash
pip install git+https://github.com/carrascomj/maudy.git
```

## Usage


```bash
 Usage: maudy [OPTIONS] COMMAND [ARGS]...

╭─ Options ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --install-completion        [bash|zsh|fish|powershell|pwsh]  Install completion for the specified shell. [default: None]                                         │
│ --show-completion           [bash|zsh|fish|powershell|pwsh]  Show completion for the specified shell, to copy it or customize the installation. [default: None]  │
│ --help                                                       Show this message and exit.                                                                         │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ──────────────────────────────────────────────╮
│ ppc      Run posterior predictive check and report it.  │
│ sample   Sample model.                                  │
╰─────────────────────────────────────────────────────────╯
```

Use `maudy sample` to run inference:

```bash
 Usage: maudy sample [OPTIONS] MAUD_DIR

 Sample model.
        
╭─ Arguments ──────────────────────────────────────────╮
│ *    maud_dir      PATH  [default: None] [required]  │
╰──────────────────────────────────────────────────────╯
╭─ Options ──────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --num-epochs                             INTEGER  [default: 100]                                                       │
│ --annealing-stage                        FLOAT    Part of training that will be annealing the KL [default: 0.2]        │
│ --normalize          --no-normalize               Whether to normalize input and output of NN [default: no-normalize]  │
│ --out-dir                                PATH     [default: None]                                                      │
│ --penalize-ss        --no-penalize-ss             [default: penalize-ss]                                               │
│ --quench             --no-quench                  [default: no-quench]                                                 │
│ --eval-flux          --no-eval-flux               [default: eval-flux]                                                 │
│ --eval-conc          --no-eval-conc               [default: eval-conc]                                                 │
│ --smoke              --no-smoke                   [default: no-smoke]                                                  │
│ --help                                            Show this message and exit.                                          │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

Use `maudy ppc` to run a Posterior Predicitve Check, where some summary statistics for variables for the different conditions will be printed:

```bash
 Usage: maudy ppc [OPTIONS] MODEL_OUTPUT

Run posterior predictive check and report it.

╭─ Arguments ───────────────────────────────────────────────╮
│ *    model_output      PATH  [default: None] [required]   │
╰───────────────────────────────────────────────────────────╯
╭─ Options ─────────────────────────────────────────────────╮
│ --num-epochs        INTEGER  [default: 800]               │
│ --help                       Show this message and exit.  │
╰───────────────────────────────────────────────────────────╯
```

### Input model

Documentation about the input model can be found at [Maud's documentation](https://maud-metabolic-models.readthedocs.io/en/latest/inputting.html).

[Maud's theory](https://maud-metabolic-models.readthedocs.io/en/latest/theory.html) applies to **maudy**, changing `MCMC` (or `NUTS`, `HMC`) for `SVI`.

**maudy** accepts the same kind of model input except for three differences:

* phosphorylation is not implemented;
* one enzyme cannot participate in more than one reaction; and
* maudy accepts an optional `maudy.toml` file.

The maudy TOML file:

```toml
# unbalanced metabolites that should be inferred by the neural
# network in the guide instead of sampled from a prior distribution 
optimize_unbalanced_metabolites = ["pi_c", "atp_c", "adp_c", "coa_c"]

# stoichiometry of ferredoxin for the reactions that contain them
[ferredoxin]
hyt = -1
hytfdr = 1
fhlrevhyt = -1
codh = 1
adr = -1
nfn = 1
rnf = -1

# hidden layer sizes for the neural networks.
[neural_network]
met_dims = [512, 1024, 1024, 1024, 512, 256]
reac_dims = [512, 512, 512, 16]
km_dims = [512, 1024, 512, 512, 256]
```

### Quenching

As aforementioned, a `--quench` parameter can be supplied to the `maudy
sample` command. If supplied, a quenching correctiong will applied to the
balanced concentrations. This quenching correction is strictly negative
(hypothesis: there is always some degree of extraction loss). The measurement
model of the fluxes and the steady-state deviation is computed using the
balanced concentrations **without** quenching. The quenched-corrected
balanced concentrations are used to compute only the measurement model of the
concentrations (hypothesis: the observed concentration come from some learnable
quenching error across conditions).

### Ferredoxin

Since ferredoxin is a prosthetic group, concepts like $k_m$ and saturation are
difficult to apply. Additionally, measuring the concentration of the oxidized
or reduced forms of ferredoxin _in vivo_ is very difficult (well, I don't know
how to do it, at least). Thus, ferredoxin is treated as a special case where
only the differences in concentration and in $\Delta G$ are used as a parameter,
such that

$$
\begin{align}
\Delta\Delta_f G_{F_d} &\sim \mathcal{LN}(77, 1) \\
\Delta [Fd] &\sim \mathcal{N}(\hat{Fd}, 0.1)
\end{align}
$$

where 77 kJ/mol arises from the difference in potential associated with transferring
2 electrons of ferredoxin. To note, not all ferredoxins transfer 2 electrons and,
depending on the enzyme environment, even the same Fe-S clusters may have different
potential differences! The ones implemented (configuration of these priors are
not implemented yet) correspond to the priors for _Clostridium autoethanogenum_.

## Examples

### Posterior Predictive checks of the examples

First, run inference on the two examples models (adapted from [Maud](https://github.com/biosustain/Maud)):

```bash
maudy sample examples/linear --num-epochs 10000 --normalize
maudy sample examples/example_ode --num-epochs 10000 --normalize
```

Each command will generate a directory of results. Rename them to `../results/
linear_actual_vae` and `../results/example_ode_actual_vae`, respectively, or
modify the notebook to point it to your result directories.

The notebook at [`notebooks/anamaudy_actual_vae.ipynb`](notebooks/
anamaudy_actual_vae.ipynb) shows how to get the approximate posterior
distributions and plot them to generate figures.

### Out-of-sample predictions of _Clostridium autoethanogenum_ on syngas

First, run inference on a model _C. autoethanogenum_ autotrophic metabolism with only two of the three conditions:

```bash
maudy sample examples/ci_aord_nosyn/ --num-epochs 100000 --annealing-stage 0.2 --normalize --quench
```
(Alternatively, the model can be run without the `--quench` parameter to remove the quenching correction.)

Again, the result directory must be renamed, in this case to `../results/cauto_results_three_cond_quench_nn_nosyngas`.
For the analysis of out-of-sample predictions of the quenched model,
see `notebooks/anamaudy_cauto_oos_quench.ipynb`.

# License

Copyright 2024, Novo Nordisk Foundation Center for Biosustainability, Technical University of Denmark.

Licensed under GNU General Public License, Version 3.0, ([LICENSE](./LICENSE)).

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, shall be licensed as above, without any additional terms or conditions.
