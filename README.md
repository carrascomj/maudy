# maudy

We have a kinetics-informed Bayesian neural network that solves accepts three kinds of variables:

1. Boundary variables.
2. Observed variables.
3. Innacessible variables.

First, the model samples the priors (boudaries of ODE).
Second, a neural network samples the innacessible variables and balanced concentrations. Third, the fluxes
are computed given all variables. Finally, the likelihood is computed for the
observed variables and, for the observed and innacessible variables that refer
to concentrations, a loss on the steady state $Sv[~Boundary] ~ 0$ is applied.

## Interface

```
╭─ Commands ───────────────────────────────────────────────╮
│ ppc      Run posterior predictive check and report it.   │
│ sample   Sample model.                                   │
╰──────────────────────────────────────────────────────────╯
```

To train a model with SVI, use `maudy sample`:


```
 Usage: maudy sample [OPTIONS] MAUD_DIR

 Sample model.

╭─ Arguments ────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *    maud_dir      PATH  [default: None] [required]                                                                    │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ──────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --num-epochs                             INTEGER  [default: 100]                                                       │
│ --annealing-stage                        FLOAT    Part of training that will be annealing the KL [default: 0.2]        │
│ --normalize          --no-normalize               Whether to normalize input and output of NN [default: no-normalize]  │
│ --out-dir                                PATH     [default: None]                                                      │
│ --penalize-ss        --no-penalize-ss             [default: penalize-ss]                                               │
│ --eval-flux          --no-eval-flux               [default: eval-flux]                                                 │
│ --eval-conc          --no-eval-conc               [default: eval-conc]                                                 │
│ --smoke              --no-smoke                   [default: no-smoke]                                                  │
│ --help                                            Show this message and exit.                                          │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

To get the posterior predictive for a trained model, use `maudy ppc`:


```
 Usage: maudy ppc [OPTIONS] MODEL_OUTPUT

 Run posterior predictive check and report it.

╭─ Arguments ────────────────────────────────────────────────╮
│ *    model_output      PATH  [default: None] [required]    │
╰────────────────────────────────────────────────────────────╯
╭─ Options ──────────────────────────────────────────────────╮
│ --num-epochs        INTEGER  [default: 800]                │
│ --help                       Show this message and exit.   │
╰────────────────────────────────────────────────────────────╯
```

## Roadmap

* [x] Model that implements a vmax-based flux model with priors only for $k_{cat}$ values and enzyme concentrations. The likelihood is computed only for the fluxes. The neural network outputs a flux correction.
* [x] Model that implements a vmax-based flux model with priors only for $k_{cat}$ values and enzyme concentrations. The likelihood is computed for fluxes and steady state. The neural network outputs a flux correction.
* [x] Model that implements all elements from the modular rate law in Maud but the regulatory terms. The likelihood is computed for fluxes, concentrations and steady state. The neural network outputs innacesible variables only referring to concentrations.
* [x] The same but with the membrane potential, the special case of ferredoxin only affecting dG and the ratio of ferredoxin being an innacessible variable.
* [] ODE with NN supplying conc_inits.
* [] Quencher.

## Debugging

* Typer generates super long tracebacks.

To disable, prefix the maudy commands with _TYPER_STANDARD_TRACEBACK=1, as in:

```bash
_TYPER_STANDARD_TRACEBACK=1 maudy sample path/to/model --normalize
```

* The loss is NaN.

Turn on jax debugging generation, prefixing maudy commands with the approapriate jax debugging env variables. For instance:

```bash
_TYPER_STANDARD_TRACEBACK=1 JAX_DEBUG_NANS=True JAX_DISABLE_JIT=True maudy sample path/to/model --num-epochs 10000 --normalize
```

This will raise an error that points to the line that generates NaNs.

* The NaN are generated at some log probability calculation, but it is unclear where.

Go to `.venv/lib/python3.12/site-packages/numpyro/infer/util.py:85` (or whatever you have it installed), and add a print statement before the logP is computed:

```python
# add this new line below before log_prob is calculated
print(f"(LP) Evaluating {site['name']}")
log_prob = site["fn"].log_prob(value)
```

This will tell us the name of the variable that generates NaNs at the logP.
