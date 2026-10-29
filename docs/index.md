---
title: BayMorph
---

<h1 align="center">
  <img alt="BayMorph" src="assets/logo/baymorph-lockup-light.svg#gh-light-mode-only" width="640">
  <img alt="BayMorph" src="assets/logo/baymorph-lockup-dark.svg#gh-dark-mode-only" width="640">
</h1>

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

    BayMorph is a work in progress (pre-1.0): its interfaces, defaults and
    output files may change between versions. It has been tested on simulated
    images, most thoroughly for the single Sérsic model, and the
    [model pages](models/index.md) state what has been validated for each
    model. No fit of a real image has yet been validated scientifically.

![Four images: an intrinsic Sérsic galaxy, the same galaxy convolved with the PSF, plus a residual background plane, and with noise added](models/figures/forward_model-light.png#gh-light-mode-only)
![Four images: an intrinsic Sérsic galaxy, the same galaxy convolved with the PSF, plus a residual background plane, and with noise added](models/figures/forward_model-dark.png#gh-dark-mode-only)

*How BayMorph sees a galaxy: the intrinsic profile, blurred by the PSF, on
top of the residual background, and observed with noise. The fit runs this
chain backwards.*

## What it does

- **Reads the header** through the conventions of the main telescopes and
  pipelines, and asks you to declare only what the header does not
  establish.
- **Prepares the image**: cuts out the target, masks the other sources,
  subtracts the sky and measures the noise.
- **Builds the PSF** from the stars of the same image.
- **Fits the model** with a likelihood that describes the noise of the sky,
  the Poisson noise of the galaxy and the correlation of neighbouring
  pixels, and samples the posterior with Nautilus, the default, or with
  Dynesty or pocoMC.
- **Reports** the posterior of every parameter, the evidence for comparing
  models, and diagnostic figures of the residuals and the radial profile.

## Where to start

<div class="grid cards" markdown>

-   **Getting Started**

    ---

    Install BayMorph from GitHub and run a first fit on a simulated galaxy.

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
- [Changelog](https://github.com/carlocannarozzophd/BayMorph/blob/main/CHANGELOG.md)
