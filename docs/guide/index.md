# User Guide

A BayMorph analysis turns one image of a galaxy into a posterior
distribution of the parameters of a morphology model. One YAML
configuration file describes the whole analysis, and one object, the
[`BayMorphRunner`](RUNNER.md), runs it.

## The workflow

| Step | What happens | Runner method | Page |
| --- | --- | --- | --- |
| 1. Configure | Write the YAML file that declares the image, the target, the model and the priors. | | [Writing a Configuration](CONFIGURING.md) |
| 2. Prepare | Cut out the target, mask the other sources, subtract the sky and measure the noise. | `prepare_image()` | [Image and Cutout](IMAGE.md), [Masks](MASKS.md), [Sky and Noise](SKY_AND_NOISE.md) |
| 3. PSF | Select isolated stars and build the point-spread function from them, or use an analytic one. | `select_psf_candidates()`, `build_psf()` | [Point-Spread Function](PSF.md) |
| 4. Fit | Sample the posterior of the model parameters with a nested sampler. | `fit()` | [Fitting](FITTING.md) |
| 5. Inspect | Read the parameter estimates and the evidence, and draw the diagnostic figures. | `diagnostics()` | [Results](RESULTS.md), [Plotting](PLOTTING.md) |

The whole sequence is one call:

```python
from baymorph import BayMorphRunner

runner = BayMorphRunner("configs/galaxy.yaml")
fit = runner.run(diagnostics=True)
```

or, from a terminal, `baymorph run configs/galaxy.yaml`. Each page of this
guide shows the same step one call at a time, with the few lines of code
that let you look at what it produced.

## How BayMorph fits a galaxy

For every set of parameter values the sampler proposes, BayMorph draws the
galaxy model, convolves it with the PSF, integrates it over the detector
pixels, adds the residual background, and compares the result with the
prepared image through the likelihood of the noise model. The sampler
explores the parameter space in proportion to likelihood times prior and
returns weighted posterior samples together with the Bayesian evidence. The
[Models](../models/index.md) section describes every model, the
[Theory](../theory/BAYESIAN_INFERENCE.md) section the statistics behind the
fit, and the [Configuration reference](../reference/CONFIGURATION.md) every
option.

## The example used in this guide

The code on these pages was run on a synthetic 1000 x 1000 px image in
counts: a Sérsic galaxy at the centre, two smaller galaxies next to it,
forty stars, a flat sky of 400 counts, Poisson noise and read noise. The
configuration was written by `baymorph config` as in the
[Quick Start](QUICK_START.md), with a fixed cutout of 251 px. Your own
image will give different numbers, but the same objects and attributes.

## Several observations and simulations

A galaxy observed in several bands, or with several instruments, is fitted
jointly with the same runner: see [Several Observations](MULTI_OBSERVATION.md).
To test a model or a prior on an image whose truth you know, see
[Simulated Images](SIMULATIONS.md).
