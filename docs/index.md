# BayMorph

<p align="center">
  <strong>Bayesian inference for galaxy morphology.</strong>
</p>

BayMorph fits models of galaxy morphology, such as a Sérsic profile or a
bulge and a disc, to astronomical images, and returns the full posterior
distribution of their parameters together with the Bayesian evidence. It
fits one image or several images of the same galaxy at once, and runs from
a single configuration file, from the command line or from Python.

```python
from baymorph import BayMorphRunner

fit = BayMorphRunner("configs/galaxy.yaml").run(diagnostics=True)
print(fit.posterior.summary("re"))
```

!!! warning "Status"

    BayMorph is under active scientific development. No fit of a real image
    has yet been validated scientifically; the order of the remaining work is
    stated in the
    [roadmap](https://github.com/carlocannarozzophd/BayMorph/blob/main/ROADMAP.md).

![Four images: an intrinsic Sérsic galaxy, the same galaxy convolved with the PSF, plus a residual background plane, and with noise added](models/figures/forward_model-light.png#gh-light-mode-only)
![Four images: an intrinsic Sérsic galaxy, the same galaxy convolved with the PSF, plus a residual background plane, and with noise added](models/figures/forward_model-dark.png#gh-dark-mode-only)

*How BayMorph sees a galaxy: the intrinsic profile, blurred by the PSF, on
top of the residual background, and observed with noise. The fit runs this
chain backwards.*

## What it does

- **Prepares the image**: cuts out the target, masks the other sources,
  subtracts the sky and measures the noise.
- **Builds the PSF** from the stars of the same image.
- **Fits the model** with a nested sampler, Nautilus by default, with a
  likelihood that describes the noise of the sky, the Poisson noise of the
  galaxy and the correlation of neighbouring pixels.
- **Reports** the posterior of every parameter, the evidence for comparing
  models, and diagnostic figures of the residuals and the radial profile.

## Where to start

<div class="grid cards" markdown>

-   **Getting Started**

    ---

    Install BayMorph and run a first fit in a few minutes.

    [Quick Start](guide/QUICK_START.md)

-   **User Guide**

    ---

    Every step of an analysis, with the code to run it and to look at what
    it produced.

    [User Guide](guide/index.md)

-   **Models**

    ---

    The galaxy models and their components, the PSF, the noise and the sky,
    with their parameters and default priors.

    [Models](models/index.md)

-   **Theory**

    ---

    Bayesian inference, nested sampling and hierarchical models for several
    observations.

    [Theory](theory/BAYESIAN_INFERENCE.md)

-   **Reference**

    ---

    Every command, every configuration key, every output file.

    [Configuration](reference/CONFIGURATION.md)

</div>

## Scope

BayMorph covers the morphology of galaxies in images, fitted to one image or
jointly to several. It does not model stellar populations, spectral energy
distributions, dust or radiative transfer, and it does not infer the
properties of galaxy populations.

## Links

- [Source code on GitHub](https://github.com/carlocannarozzophd/BayMorph)
- [Roadmap](https://github.com/carlocannarozzophd/BayMorph/blob/main/ROADMAP.md)
- [Changelog](https://github.com/carlocannarozzophd/BayMorph/blob/main/CHANGELOG.md)
- [For developers](DEVELOPERS.md)
