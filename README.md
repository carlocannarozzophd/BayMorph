# BayMorph

<p align="center">
  <strong>Bayesian structural decomposition of galaxies from astronomical images</strong>
</p>

<p align="center">
  <img alt="Version" src="https://img.shields.io/badge/version-v0.1.0-blue">
  <img alt="Python" src="https://img.shields.io/badge/python-3.11%2B-green">
  <img alt="Tests" src="https://img.shields.io/badge/tests-159%2F159%20passed-brightgreen">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-lightgrey">
  <img alt="Status" src="https://img.shields.io/badge/status-active%20development-orange">
</p>

> [!IMPORTANT]
> **Status:** BayMorph **v0.1.0**. The public workflow, notebooks, documentation, and automated tests are available. Scientific validation on larger real-galaxy samples is ongoing.

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
Model comparison
    ↓
Diagnostics & reports
```

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

config = load_resolved_config("configs/simulated_galaxy.yaml")

runner = BayMorphRunner(config)
results = runner.run()
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

- Version **0.1.0**
- **159 / 159** automated tests passing
- Active development
- MIT License
