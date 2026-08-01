# BayMorph

<p align="center">
  <strong>Bayesian forward modelling for galaxy morphology.</strong>
</p>

<p align="center">
  <a href="https://github.com/carlocannarozzophd/BayMorph/releases"><img alt="Version" src="https://img.shields.io/badge/version-v0.4-blue"></a>
  <img alt="Python" src="https://img.shields.io/badge/python-%E2%89%A53.11-green">
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-MIT-lightgrey"></a>
  <a href="https://github.com/carlocannarozzophd/BayMorph/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/carlocannarozzophd/BayMorph/actions/workflows/ci.yml/badge.svg"></a>
  <img alt="Status" src="https://img.shields.io/badge/status-active%20development-orange">
</p>

<p align="center">
  <a href="#installation">Installation</a> ·
  <a href="#quick-start">Quick Start</a> ·
  <a href="notebooks/01_learn_baymorph.ipynb">Tutorial</a> ·
  <a href="docs/API.md">API</a> ·
  <a href="docs/ARCHITECTURE.md">Architecture</a> ·
  <a href="docs/VALIDATION.md">Validation</a>
</p>

---

> **Need galaxy morphology?**
>
> **BayMorph.**
>
> Need Bayesian inference?
>
> **BayMorph.**
>
> Need a relaxing afternoon by the bay?
>
> Sorry.
> We only borrowed the name.

---

> [!IMPORTANT]
>
> BayMorph is under active scientific development.
> The current implementation provides a physically consistent forward model,
> Bayesian inference through nested sampling,
> evidence-based model comparison,
> reproducible image preparation,
> posterior diagnostics,
> and an automated scientific validation framework.

# What is BayMorph?

BayMorph is an open-source scientific Python framework for Bayesian galaxy
morphology.

Unlike traditional profile-fitting software, BayMorph does **not** consider
model evaluation as the direct sampling of an analytical function on detector
pixels.

Instead, BayMorph models the **complete image-formation process**.

A galaxy is treated as a continuous distribution of light.

The detector records neither the analytical profile nor a collection of point
samples.

It records pixel-integrated flux after convolution with the instrumental
point-spread function.

BayMorph explicitly models every stage of this process before evaluating the
likelihood.

The result is a statistically rigorous Bayesian inference framework where
posterior distributions, parameter degeneracies and Bayesian evidence are
first-class scientific products rather than optional post-processing outputs.

> **Galaxy morphology is fundamentally an inference problem, not merely an optimization problem.**

---

# Scientific Motivation

Recovering the structural properties of galaxies is one of the central problems
of observational astronomy.

Parameters such as

- total luminosity;
- effective radius;
- Sérsic index;
- projected axis ratio;
- position angle;
- centroid;

are widely used to investigate galaxy formation and evolution.

However, these parameters are never observed directly.

Instead, observations are affected by

- atmospheric and instrumental PSFs;
- detector pixel integration;
- finite spatial sampling;
- noise;
- background estimation;
- masking;
- parameter degeneracies.

Consequently, galaxy morphology is naturally formulated as a Bayesian inverse
problem rather than a deterministic optimization problem.

BayMorph has been designed around this philosophy from the beginning.

---

# Bayesian Forward Modelling

The current forward model implemented by BayMorph follows the physical image
formation process.

```text
Physical parameters
        │
        ▼
Continuous Sérsic light distribution
        │
        ▼
Analytic flux normalization
        │
        ▼
Detector pixel integration
        │
        ▼
Rendering on an extended domain
        │
        ▼
PSF convolution
        │
        ▼
Observed detector footprint
        │
        ▼
Likelihood evaluation
        │
        ▼
Posterior distribution
```

This implementation guarantees that

- total flux is defined on the continuous profile;
- the effective radius preserves its physical half-light definition;
- rendered pixels are invariant under cutout enlargement;
- detector pixels measure integrated flux rather than point samples;
- PSF convolution includes light scattered into the observed footprint from
  outside the detector cutout.

These properties are essential for future simultaneous inference from multiple
observations of the same galaxy.

---

# Why BayMorph?

Most morphology workflows combine multiple independent tools for

- image preparation;
- source detection;
- masking;
- PSF estimation;
- model fitting;
- posterior analysis;
- diagnostic visualization.

Besides increasing operational complexity, this often makes scientific
reproducibility difficult.

BayMorph integrates the complete workflow inside a single coherent scientific
framework.

Scientific algorithms remain modular and independently testable.

Configuration files completely define an analysis.

Notebooks become lightweight user interfaces rather than hidden implementations.

---

# Main Capabilities

## Physically motivated forward model

- Continuous Sérsic light distributions.
- Analytic total-flux normalization.
- Continuous half-light radius definition.
- Detector pixel integration.
- Rendering on extended coordinate domains.
- Physically consistent PSF convolution.
- Single- and double-Sérsic models.
- Composite morphological models.

## Bayesian inference

- Dynesty nested sampling.
- Bayesian evidence.
- Bayes factors.
- Posterior summaries.
- Equal-weight posterior samples.
- Configurable priors.
- Gaussian likelihoods.
- Cash and Poisson-Gaussian likelihood infrastructure.

## Image preparation

- FITS loading.
- Target identification.
- Automatic and fixed cutouts.
- Background estimation.
- Noise estimation.
- Connected-source masking.
- Optional mask refinement.
- Target geometry estimation.
- Elliptical radial profiles.
- Plateau diagnostics.

## PSF workflows

- Analytic Gaussian PSFs.
- Empirical PSFs.
- Candidate selection.
- Candidate recentering.
- Candidate validation.
- PSF stacking.
- PSF persistence.
- PSF normalization.

---

## Diagnostics

BayMorph diagnostics are designed to expose the statistical structure of the
inference rather than only the visual quality of the reconstruction.

Available diagnostics include

- posterior corner plots;
- posterior radial-profile bands;
- model images;
- residual maps;
- normalized residuals;
- radial surface-brightness profiles;
- image-preparation reports;
- reconstruction reports;
- runtime reports.

A visually good residual map does **not** imply that every parameter is well
constrained.

Posterior geometry is considered part of the scientific result.

---

# Installation

BayMorph currently supports **Python 3.11 and later**.

A dedicated Conda environment is recommended.

## Install from GitHub

```bash
conda create -n baymorph -c conda-forge python=3.11 -y
conda activate baymorph

python -m pip install --upgrade pip

python -m pip install \
  "git+https://github.com/carlocannarozzophd/BayMorph.git@main"
```

Verify the installation

```bash
baymorph --version
baymorph --help
```

---

## Development installation

```bash
git clone https://github.com/carlocannarozzophd/BayMorph.git

cd BayMorph

python -m pip install --upgrade pip

python -m pip install -e ".[dev]"
```

Run the quality checks

```bash
git diff --check

python -m ruff check .

python -m pytest -q
```

---

# Quick Start

BayMorph analyses are driven by YAML configuration files.

```python
from baymorph import BayMorphRunner

runner = BayMorphRunner(
    "configs/learn_baymorph.yaml",
)

runner.prepare_image()
runner.fit()
runner.diagnostics()
```

The principal products remain accessible from the runner

```python
prepared_image = runner.image

posterior = runner.results

profile_bands = runner.profile_bands
```

For empirical PSF workflows

```python
runner.prepare_image()

runner.select_psf_candidates()

runner.build_psf()

runner.fit()

runner.diagnostics()
```

---

# Learn BayMorph

The recommended starting point is

```
notebooks/01_learn_baymorph.ipynb
```

The tutorial demonstrates

1. YAML configuration
2. synthetic galaxy generation
3. image preparation
4. background estimation
5. noise estimation
6. Gaussian PSF construction
7. Bayesian fitting
8. posterior interpretation
9. scientific diagnostics

---

# Scientific Workflow

A typical analysis proceeds as

```text
Observation
      │
      ▼
Target identification
      │
      ▼
Cutout construction
      │
      ▼
Background estimation
      │
      ▼
Noise estimation
      │
      ▼
Mask construction
      │
      ▼
PSF construction
      │
      ▼
Physical forward model
      │
      ▼
Likelihood
      │
      ▼
Nested Sampling
      │
      ▼
Posterior
      │
      ▼
Scientific diagnostics
```

Every processing step is explicitly configured.

Nothing is hidden inside notebooks.

Resolved configurations together with all scientific products are written into
a structured run directory to guarantee reproducibility.

---

# Public API

The stable top-level API currently exports

```python
from baymorph import (
    BayMorphRunner,
    CompositeModel,
    ConvolvedModel,
    DoubleSersicModel,
    EmpiricalPSF,
    GaussianPSF,
    ImageData,
    Observation,
    ObservationMetadata,
    PosteriorResult,
    RealFitProducts,
    SersicModel,
    SyntheticObservation,
    SyntheticSersic,
)
```

See

```
docs/API.md
```

for the complete API documentation.

---

# Architecture

BayMorph follows a layered architecture.

```text
User code
      │
      ▼
BayMorphRunner
      │
      ▼
Pipeline
      │
      ├── image preparation
      ├── PSF workflows
      ├── fitting
      └── diagnostics
      │
      ▼
Scientific modules
      │
      ├── observations
      ├── models
      ├── rendering
      ├── likelihoods
      ├── samplers
      ├── simulations
      ├── validation
      └── results
```

The rendering API preserves physical image coordinates through

```python
model.render(
    shape=(ny, nx),
    origin=(y0, x0),
)
```

which allows the same intrinsic model to be rendered over arbitrary detector
footprints without changing already-rendered pixels.

The complete design is documented in

```
docs/ARCHITECTURE.md
```


---

# Repository Layout

```text
BayMorph/
├── baymorph/
│   ├── core/
│   ├── fitting/
│   ├── models/
│   ├── pipeline/
│   ├── plotting/
│   ├── resources/
│   ├── results/
│   ├── simulation/
│   ├── validation/
│   └── runner.py
│
├── configs/
├── docs/
├── notebooks/
├── tests/
│
├── CHANGELOG.md
├── CITATION.cff
├── CONTRIBUTING.md
├── DECISIONS.md
├── ROADMAP.md
└── pyproject.toml
```

Every scientific algorithm lives inside the Python package.

Documentation, architectural notes and research material live under
`docs/`.

---

# Validation

Scientific software should not merely execute.

It should demonstrate that its scientific assumptions are internally
consistent.

BayMorph therefore validates the forward model independently at multiple
levels.

## Forward-model validation

Current validation includes

- analytic Sérsic flux normalization;
- continuous half-light radius consistency;
- detector-pixel integration;
- cutout invariance;
- rendering consistency across coordinate systems;
- PSF normalization;
- finite-domain PSF convolution;
- consistency between simulation and inference renderers.

## Inference validation

The Bayesian layer is validated through

- synthetic recovery experiments;
- posterior consistency;
- parameter-degeneracy studies;
- Bayesian evidence regression;
- workflow regression tests;
- independent cross-checks.

## Software validation

The automated suite currently validates

- public API;
- configuration system;
- CLI;
- image preparation;
- masking;
- PSFs;
- rendering;
- samplers;
- likelihoods;
- posterior summaries;
- diagnostics;
- packaged resources.

Validation is considered an integral part of the scientific methodology
rather than a software engineering afterthought.

See

```
docs/VALIDATION.md
```

for the complete validation framework.

---

# Current Status

## Implemented

✔ Continuous Sérsic profiles

✔ Analytic flux normalization

✔ Physical half-light radius definition

✔ Detector-pixel integration

✔ Rendering on extended coordinate domains

✔ Physically consistent PSF convolution

✔ Gaussian and empirical PSFs

✔ Single- and double-Sérsic models

✔ Bayesian evidence

✔ Dynesty backend

✔ Configuration-driven workflows

✔ Posterior diagnostics

✔ Scientific validation framework

---

## Current priorities

The next scientific milestones are

1. Adaptive pixel integration.

2. Rendering acceleration.

3. Cached Fourier-domain PSF convolution.

4. Multi-observation Bayesian inference.

5. Shared-parameter inference.

6. Hierarchical Bayesian inference.

The long-term objective is simultaneous Bayesian inference from multiple
correlated observations of the same astronomical object.

---

# Documentation

Additional documentation is available in

- `docs/API.md`
- `docs/ARCHITECTURE.md`
- `docs/VALIDATION.md`
- `docs/README.md`
- `ROADMAP.md`
- `DECISIONS.md`
- `CHANGELOG.md`

---

# Contributing

Contributions are welcome.

Before every commit, execute

```bash
git diff --check

python -m ruff check .

python -m pytest -q
```

Changes affecting

- forward models;
- likelihoods;
- priors;
- rendering;
- inference;
- posterior summaries;

must include corresponding scientific validation and synchronized
documentation.

Development discussions and architectural decisions should be recorded in
the appropriate documentation rather than remaining implicit inside pull
requests.

See

```
CONTRIBUTING.md
```

for the complete development workflow.

---

# Citation

Citation metadata are provided in

```
CITATION.cff
```

When using development versions of BayMorph in scientific work,
please record

- the Git commit;
- the resolved YAML configuration;
- the BayMorph version.

---

# License

BayMorph is distributed under the MIT License.

See

```
LICENSE
```

for details.

