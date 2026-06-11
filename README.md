# BayMorph

Bayesian morphological modelling of astronomical images.

BayMorph is a lightweight Python framework for forward modelling galaxy images using Bayesian inference. The project is designed around physically motivated image models, realistic observational effects, and posterior-based parameter estimation.

The current focus is single-Sérsic galaxy fitting, with support for synthetic observations, FITS-based workflows, PSF convolution, nested sampling, posterior analysis, and real-data validation.

---

# Features

## Simulation

- Synthetic Sérsic galaxy generation
- Gaussian PSF generation
- PSF convolution
- Synthetic observation generation
- Gaussian noise injection
- Poisson noise injection

## FITS and Data Handling

- FITS image loading
- FITS image writing
- Observation containers
- Image metadata preservation

## Background and Noise

- Iterative sigma-clipped background estimation
- Photutils Background2D estimation
- Source-masked background estimation
- Constant noise maps
- Empirical noise-map estimation

## Source Detection and Masking

- Automatic source detection
- Binary source masks
- Masked background estimation workflows

## Models

- Sérsic surface-brightness model
- PSF-convolved models via `ConvolvedModel`

## Bayesian Inference

- Grid-search fitting
- Flux-only Dynesty fitting
- Flux + effective-radius Dynesty fitting
- Flux + effective-radius + Sérsic-index Dynesty fitting
- Generic Dynesty fitting
- Parameter framework (`Parameter`, `ParameterSet`)
- Optional PSF-aware likelihood evaluation
- Optional masking support

## Posterior Analysis

- Posterior summaries
- Weighted posterior statistics
- Median parameter estimates
- Best-likelihood parameter estimates
- Posterior sample storage
- Dynesty posterior reconstruction

## Plotting

- 1D posterior plots
- 2D posterior plots
- Corner plots
- Data-model-residual plots
- Unified plotting style utilities

---

# Installation

Clone the repository:

```bash
git clone <repository-url>
cd BayMorph
```

Create an environment:

```bash
conda create -n baymorph python=3.11
conda activate baymorph
```

Install:

```bash
pip install -e .
```

Install development dependencies:

```bash
pip install -e ".[dev]"
```

---

# Quick Example

```python
from baymorph import (
    DynestyFitter,
    SersicModel,
    make_constant_noise_map,
)

data = SersicModel(
    flux=1000,
    re=8,
    n=2,
    q=0.8,
    theta=0.0,
    x0=30,
    y0=30,
).render(
    shape=(60, 60),
)

noise = make_constant_noise_map(
    data.shape,
    sigma=1.0,
)


def model_factory(
    flux,
    re,
    n,
):
    return SersicModel(
        flux=flux,
        re=re,
        n=n,
        q=0.8,
        theta=0.0,
        x0=30,
        y0=30,
    )


fitter = DynestyFitter(
    model_factory=model_factory,
    data=data,
    noise=noise,
    parameter_names=[
        "flux",
        "re",
        "n",
    ],
    prior_bounds={
        "flux": (100, 2000),
        "re": (2, 20),
        "n": (0.5, 8),
    },
)

result = fitter.run(
    nlive=100,
)

print(
    result.best_parameters
)
```

---

# Current Status

BayMorph is currently a working MVP for Bayesian single-Sérsic fitting.

Implemented:

- FITS workflows
- Background estimation
- Source masking
- Noise estimation
- Sérsic modelling
- PSF convolution
- Dynesty-based inference
- Posterior analysis
- Real FITS validation

Recent milestones:

- First real FITS ingestion
- First real single-Sérsic fit
- First PSF-convolved real fit
- Convolved model abstraction
- Parameter framework
- ParameterSet-aware Dynesty fitting

---

# Validation

Current automated test suite:

```text
50 passing tests
```

Coverage includes:

- Background estimation
- Photutils workflows
- Source masking
- FITS I/O
- Noise handling
- PSF convolution
- Dynesty fitting
- Posterior analysis
- Parameter framework
- Convolved models

---

# Roadmap

## Near-term

- Real-data fitting pipeline
- Automatic target extraction
- Improved masking workflows
- Empirical PSF estimation
- Double-Sérsic model
- Bulge + disk decomposition

## Mid-term

- Multi-component galaxy models
- Hierarchical parameter management
- Real survey pipelines
- JWST and HST workflows

## Long-term

- Population-scale Bayesian morphology
- Automated catalogue production
- Large-survey deployment

---

# License

MIT License.