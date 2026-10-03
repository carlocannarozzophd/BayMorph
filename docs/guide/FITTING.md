# Fitting

```python
fit = runner.fit()
```

`fit()` samples the posterior of the model parameters given the prepared
image, the PSF, the priors and the noise model, and writes the results to
`fits/<model>/` in the run directory. This page covers the choices that
define a fit: the model, the priors and the sampler.

## The model

```yaml
fit:
  models: [single_sersic]
```

`fit.models` lists the morphology models to fit; each is fitted separately,
in its own directory. With several models, `fit()` returns a dictionary of
fit products by model name. The available models are:

| Model | Describes |
| --- | --- |
| [`single_sersic`](../models/SINGLE_SERSIC.md) | one Sérsic profile: the baseline description of a galaxy |
| [`double_sersic`](../models/DOUBLE_SERSIC.md) | two Sérsic profiles with a common centre |
| [`sersic_point_source`](../models/SERSIC_POINT_SOURCE.md) | a Sérsic profile and an unresolved nucleus |
| [`bulge_disk`](../models/BULGE_DISK.md) | a Sérsic bulge and an exponential disc |
| [`boxy_sersic`](../models/BOXY_SERSIC.md) | a Sérsic profile with boxy or discy isophotes |
| [`core_sersic`](../models/CORE_SERSIC.md) | a Sérsic profile with a depleted core |
| [`bulge_broken_disk`](../models/BULGE_BROKEN_DISK.md) | a bulge and a disc whose slope changes at a break radius |
| [`bulge_disk_bar`](../models/BULGE_DISK_BAR.md) | a bulge, a disc and a bar |
| [`bulge_disk_ring`](../models/BULGE_DISK_RING.md) | a bulge, a disc and a ring |
| [`bulge_disk_spiral`](../models/BULGE_DISK_SPIRAL.md) | a bulge and a disc with two spiral arms |

Each page of the [Models](../models/index.md) section lists the parameters
of the model, their units and their default priors.

## The priors

Every sampled parameter has a prior. The defaults are wide uniform ranges in
pixels, except for the fluxes, which have none and must always be declared:

```yaml
fit:
  priors:
    single_sersic:
      flux: {type: log_uniform, lower: 1.0e3, upper: 1.0e7}
      re: {type: uniform, lower: 2, upper: 60}       # pixels
      n: {type: uniform, lower: 0.5, upper: 6}
      theta: {type: uniform, lower: -pi/2, upper: pi/2}
      x0: {type: uniform, lower: -5, upper: 5, relative_to: target_x}
      y0: {type: uniform, lower: -5, upper: 5, relative_to: target_y}
```

| Type | Keys |
| --- | --- |
| `uniform` | `lower`, `upper` |
| `log_uniform` | `lower`, `upper`; uniform in the logarithm, for a scale such as a flux or a radius spanning decades |
| `normal` | `mean`, `sigma` |
| `beta` | `alpha`, `beta`, and optionally `lower` and `upper` |

Angles can be written with `pi`. `relative_to: target_x` or `target_y`
centres a uniform prior on the target position, so `x0` above may lie
within 5 pixels of the position given in `data.target_position`. Parameters
not listed keep their defaults, which every model page gives.

The residual background has its own priors, under `fit.background`: see
[Sky and Noise](SKY_AND_NOISE.md).

## The noise model

```yaml
fit:
  likelihood:
    type: correlated_gaussian
    effective_gain:
      source: measured_sky
```

The likelihood is a Gaussian that describes the noise of the sky, the
Poisson noise of the galaxy and the correlation of neighbouring sky pixels.
The only choice it asks of you is the effective gain, explained in
[Sky and Noise](SKY_AND_NOISE.md).

## The sampler

```yaml
fit:
  sampler:
    name: nautilus          # nautilus, dynesty or pocomc
    execution:
      processes: auto       # number of parallel processes
    init:                   # options of the sampler, with their native names
      n_live: 2000
      seed: 42
      filepath: auto        # keep a checkpoint, so an interrupted fit resumes
    run:
      n_eff: 10000
```

| Sampler | What it is |
| --- | --- |
| `nautilus` (default) | [Nautilus](https://nautilus-sampler.readthedocs.io/), nested sampling boosted by neural networks |
| `dynesty` | [Dynesty](https://dynesty.readthedocs.io/), static or, with `mode: dynamic`, dynamic nested sampling |
| `pocomc` | [pocoMC](https://pocomc.readthedocs.io/), preconditioned Monte Carlo with normalizing flows |

The options under `init` and `run` are passed unchanged to the sampler, with
the names and defaults of its own documentation: for Nautilus, `n_live`
live points, 2000 by default, and `n_eff`, the effective number of
posterior samples at which it stops, 10000 by default. BayMorph owns only
the likelihood, the priors and the parameters, which cannot be overridden
there. `processes: auto` uses the available processors minus two, at most
ten. A `seed` makes the run reproducible.

All three samplers return the same result: weighted posterior samples and
the Bayesian evidence. The choice of sampler changes how the posterior is
computed, never the scientific problem. [Nested Sampling](../theory/NESTED_SAMPLING.md)
explains how they work.

### How long a fit takes

The cost grows with the number of pixels in the cutout and with the number
of parameters. On an Apple M2 Pro with ten processes, the simulated example
of the [Quick Start](QUICK_START.md) takes about a minute and a half, and a
single Sérsic fit of a 251 x 251 px cutout with an empirical PSF took about
four minutes with `n_live: 300`. A first look with fewer live points is a good way to
check a configuration before a full run.

## Comparing models

The evidence of each fit measures how well its model explains the image,
penalized by the volume of its prior. Two models fitted to the same
prepared image, with the same mask, PSF and noise model, can be compared by
the difference of their log-evidences, the logarithm of the Bayes factor.
With `fit.comparison.enabled: true` and `fit.models: [single_sersic,
double_sersic]`, BayMorph computes that Bayes factor and writes it to
`model_comparison/bayes_factor.json`. BayMorph ranks models; it never picks
one for you. [Bayesian Inference](../theory/BAYESIAN_INFERENCE.md#the-evidence-and-model-comparison)
explains how to read a Bayes factor.
