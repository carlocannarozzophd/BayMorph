# BayMorph

<p align="center">
  <strong>Bayesian structural decomposition of galaxies from astronomical images</strong>
</p>

<p align="center">
  <img alt="Version" src="https://img.shields.io/badge/version-v0.2.0-blue">
  <img alt="Python" src="https://img.shields.io/badge/python-3.11%2B-green">
  <img alt="Tests" src="https://img.shields.io/badge/tests-157%2F157%20passed-brightgreen">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-lightgrey">
  <img alt="Status" src="https://img.shields.io/badge/status-active%20development-orange">
</p>

> [!IMPORTANT]
> **Status:** BayMorph **v0.2.0**. BayMorph now provides a model-agnostic Bayesian morphology framework supporting composite galaxy models through a common rendering interface. Scientific validation on large observational samples is ongoing.

BayMorph is an open-source Python framework for **Bayesian structural decomposition of galaxies**. It provides a complete, reproducible workflow from astronomical FITS images to structural modelling, Bayesian model comparison, and publication-quality diagnostics.

---

## Why BayMorph?

BayMorph is designed around a **workflow-first philosophy**: instead of providing only a galaxy fitting routine, it manages the complete analysis from the input image to the final scientific products.

| Want to... | BayMorph helps you... |
| --- | --- |
| Fit a galaxy from a FITS image | Run the complete workflow from image preparation to Bayesian inference. |
| Generate realistic mock galaxies | Create simulated galaxies and observations with known input parameters. |
| Analyse real observations | Prepare FITS images, estimate background and noise, construct or load a PSF, and fit galaxy models. |
| Compare competing models | Use Bayesian evidence and Bayes factors for principled model selection. |
| Analyse many galaxies | Process entire directories of FITS images using the same configuration template. |
| Produce publication-quality figures | Automatically generate residual maps, corner plots, radial profiles, and summary reports. |
| Develop new galaxy models | Implement `render()` and `render_components()` and reuse the existing Bayesian inference, diagnostics, and plotting pipeline. |

---

## Features

- Bayesian nested sampling using Dynesty
- Bayesian model comparison through evidence and Bayes factors
- Composite morphological models
- Model-agnostic rendering interface
- Automatic posterior-profile computation
- Publication-quality diagnostics
- Simulated and real-galaxy workflows
- Batch processing
- Reproducible configuration-driven pipeline
- Configuration-driven workflows through YAML templates

---

## Workflow

```text
FITS image
    ↓
Image preparation
    ↓
Background & noise estimation
    ↓
Source detection & masking
    ↓
PSF construction / loading
    ↓
Bayesian fitting
    ↓
Posterior profiles
    ↓
Model comparison
    ↓
Diagnostics & reports
```

---

## Model architecture

BayMorph is built around a model-agnostic rendering interface.

Every morphological model implements the common rendering interface

- `render(shape)`
- `render_components(shape)`

where

```python
model.render(shape)
```

returns the complete galaxy model image, while

```python
model.render_components(shape)
```

returns a dictionary of named structural components, for example

```python
{
    "bulge": ...,
    "disk": ...,
}
```

or

```python
{
    "bulge": ...,
    "bar": ...,
    "disk": ...,
}
```

This interface allows the Bayesian inference pipeline, posterior-profile computation, diagnostics, and plotting routines to work with arbitrary composite galaxy models without model-specific logic.

---

## Installation

```bash
git clone https://github.com/carlocannarozzophd/BayMorph.git
cd BayMorph

conda create -n baymorph python=3.11
conda activate baymorph

pip install -e .
```

For development:

```bash
pip install -e ".[dev]"
```

---

## Quick start

```python
from baymorph import BayMorphRunner, load_resolved_config

config = load_resolved_config(
    "configs/simulated_galaxy.yaml"
)

runner = BayMorphRunner.from_config(config)

runner.prepare_image()
runner.select_psf_candidates()
runner.build_psf()

results = runner.fit(return_result=True)

runner.diagnostics()
```

---

## Official notebooks

| Notebook | Purpose |
| --- | --- |
| [`01_Simulated_Galaxy_Analysis.ipynb`](notebooks/01_Simulated_Galaxy_Analysis.ipynb) | Complete workflow on a simulated galaxy. |
| [`02_Real_Galaxy_Analysis.ipynb`](notebooks/02_Real_Galaxy_Analysis.ipynb) | Complete workflow on a real galaxy. |
| [`03_Multi_Galaxy_Analysis.ipynb`](notebooks/03_Multi_Galaxy_Analysis.ipynb) | Batch analysis of multiple FITS images. |

---

## Documentation

### Tutorials

- [`01_Simulated_Galaxy_Analysis.ipynb`](notebooks/01_Simulated_Galaxy_Analysis.ipynb)
- [`02_Real_Galaxy_Analysis.ipynb`](notebooks/02_Real_Galaxy_Analysis.ipynb)
- [`03_Multi_Galaxy_Analysis.ipynb`](notebooks/03_Multi_Galaxy_Analysis.ipynb)

### Technical documentation

- [`docs/README.md`](docs/README.md)
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
- [`docs/VALIDATION.md`](docs/VALIDATION.md)

### Project documents

- [`DECISIONS.md`](DECISIONS.md)
- [`ROADMAP.md`](ROADMAP.md)

---

## Current status

- Version **0.2.0**
- Composite-model architecture
- Model-agnostic posterior-profile pipeline
- Independent component geometry for Double Sérsic models
- **157 / 157** automated tests passing
- Active development
- MIT License
