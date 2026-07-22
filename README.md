# BayMorph

<p align="center">
  <strong>A modern Bayesian framework for Bayesian galaxy morphology modelling from astronomical imaging.</strong>
</p>

<p align="center">
  <img alt="Version" src="https://img.shields.io/badge/version-v0.2.0-blue">
  <img alt="Python" src="https://img.shields.io/badge/python-%E2%89%A53.10-green">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-lightgrey">
  <img alt="Status" src="https://img.shields.io/badge/status-active%20development-orange">
</p>

> [!IMPORTANT]
> BayMorph is under active scientific development. Version 0.2.0 provides an end-to-end, configuration-driven workflow for image preparation, Bayesian Sérsic fitting, model comparison, and scientific diagnostics. Validation on heterogeneous observational datasets is ongoing.

BayMorph is an open-source Python package for **Bayesian structural analysis of galaxies**.

Starting from astronomical FITS images, BayMorph performs

* automatic image preparation;
* adaptive target identification and cutout generation;
* empirical PSF construction;
* Bayesian Sérsic model fitting;
* Bayesian model comparison;
* posterior analysis;
* publication-quality diagnostics;

through a **fully reproducible, configuration-driven workflow**.

The project is built as a layered scientific software library in which notebooks remain lightweight while scientific algorithms, workflow orchestration, diagnostics, and output management are implemented inside reusable, tested package modules.

---

# Why BayMorph?

Traditional galaxy-fitting workflows often require combining multiple independent tools for image preparation, source detection, PSF construction, Bayesian inference, visualization, and diagnostics.

BayMorph unifies these tasks into a single scientific framework driven by YAML configuration files rather than notebook-specific logic. This architecture improves reproducibility, minimizes boilerplate code, and keeps scientific algorithms independent of user interfaces.

The project is designed around a simple principle:

> **Scientific algorithms should be reusable, reproducible, and independent from notebook logic.**

---

# Key Features

* End-to-end Bayesian galaxy morphology workflow.
* Reproducible YAML-based configuration.
* Automatic target-specific image preparation.
* Adaptive cutout sizing from galaxy light profiles.
* Empirical PSF candidate selection and stacking.
* Bayesian nested sampling with Dynesty.
* Bayesian evidence and model comparison.
* Publication-quality diagnostics and reports.

---

# Architecture

BayMorph follows a layered scientific architecture in which every package owns a single, well-defined responsibility.

```text
simple notebook
        │
        ▼
BayMorphRunner
        │
        ▼
pipeline orchestration
        │
        ▼
scientific core modules
```

A complete description of the software architecture is available in:

* [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)

---

# Current Capabilities

## Image Preparation

* FITS image preparation.
* Automatic target identification and segmentation.
* Fixed and adaptive cutout generation.
* Source masking.
* Elliptical radial-profile construction.
* Plateau-based cutout sizing.
* Target-specific sky-region estimation.
* Robust constant-background estimation.
* MAD-based noise estimation.

## PSF Modelling

* PSF loading.
* Empirical PSF candidate selection.
* Robust PSF stacking.

## Galaxy Models

* Single Sérsic models.
* Double Sérsic models.
* PSF-convolved rendering.
* Model-agnostic rendering interface.

## Bayesian Inference

* Gaussian image likelihoods.
* Dynesty nested sampling.
* Configurable convergence criteria.
* Parallel execution.
* Bayesian evidence computation.
* Bayes-factor model comparison.

## Diagnostics

* Corner plots.
* Residual maps.
* Posterior radial-profile bands.
* Reconstruction reports.
* Image-preparation reports.

---

# Automatic Image Preparation

For `data.cutout.mode: auto`, BayMorph performs the following workflow:

```text
full image
      ↓
target detection
      ↓
adaptive cutout
      ↓
target geometry
      ↓
radial profile
      ↓
plateau detection
      ↓
sky estimation
      ↓
background and noise
      ↓
Bayesian fitting
      ↓
diagnostics
```

The configured `target_position` is used only as an initial seed for identifying the target. Once the final segmentation has been completed, all subsequent measurements use the measured target geometry.

The legacy fixed-size workflow remains available when required.

Complete implementation details are documented in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

# Installation

BayMorph is designed to be installed inside a dedicated Conda environment.

## 1. Create a Conda environment

```bash
conda create -n baymorph -c conda-forge python=3.11 -y
conda activate baymorph
```

## 2. Upgrade pip

```bash
python -m pip install --upgrade pip
```

## 3. Install BayMorph

```bash
python -m pip install \
"git+https://github.com/carlocannarozzophd/BayMorph.git@main"
```

## 4. Verify the installation

```bash
baymorph --version
baymorph --help
```

BayMorph supports **Python 3.10 and later**.

---

# Quick Start

```python
from baymorph import BayMorphRunner, load_resolved_config

config = load_resolved_config("configs/simulated_galaxy.yaml")

runner = BayMorphRunner.from_config(config)

runner.prepare_image()
runner.fit()
runner.diagnostics()

result = runner.results

print(result.best_parameters)
print(result.statistics)
```

Workflows requiring an empirical PSF simply extend the pipeline:

```python
runner.prepare_image()
runner.select_psf_candidates()
runner.build_psf()
runner.fit()
runner.diagnostics()
```

Pipeline products are exposed through

```python
runner.image
runner.psf_candidates
runner.psf
runner.results
```

Each workflow stage can optionally return its products directly:

```python
products = runner.prepare_image(return_result=True)
fit_result = runner.fit(return_result=True)
```

---

# Command-Line Interface

Check the installed version:

```bash
baymorph --version
```

Create a new external project:

```bash
baymorph init MyBayMorphProject
```

This creates

```text
MyBayMorphProject/
├── configs/
├── notebooks/
├── data/
└── runs/
```

Copy only the packaged examples:

```bash
baymorph examples BayMorphExamples
```

Use

```bash
--force
```

to overwrite existing files.

---

# Updating BayMorph

Update to the latest development version:

```bash
conda activate baymorph

python -m pip install \
    --upgrade \
    --force-reinstall \
    "git+https://github.com/carlocannarozzophd/BayMorph.git@main"
```

Refresh packaged notebooks and configuration files:

```bash
cd MyBayMorphProject

baymorph examples . --force
```

This updates

```text
configs/
notebooks/
```

while preserving

```text
data/
runs/
```

and all scientific outputs.

---

# Official Notebooks

| Notebook                             | Purpose                                      |
| ------------------------------------ | -------------------------------------------- |
| `01_Simulated_Galaxy_Analysis.ipynb` | End-to-end validation on simulated galaxies. |
| `02_Real_Galaxy_Analysis.ipynb`      | Bayesian analysis of a real galaxy.          |
| `03_Multi_Galaxy_Analysis.ipynb`     | Batch processing workflow.                   |

---

# Model Architecture

Every morphology model implements

```python
model.render(shape)

model.render_components(shape)
```

This common interface keeps fitting, diagnostics, plotting, and posterior analysis independent of individual morphology models.

---

# Flux Convention

The current discrete Sérsic renderer normalizes the requested flux over the finite rendered image.

Consequently, the `flux` parameter corresponds to the total flux represented inside the rendered array rather than the analytic infinite-radius Sérsic integral.

Simulation and fitting therefore remain internally consistent. A future renderer may introduce validated analytic pixel-integrated normalization as an optional alternative.

---

# Documentation

| Document               | Purpose                          |
| ---------------------- | -------------------------------- |
| `README.md`            | Project overview and quick start |
| `docs/ARCHITECTURE.md` | Software architecture            |
| `docs/README.md`       | Documentation index              |
| `docs/VALIDATION.md`   | Scientific validation            |
| `DECISIONS.md`         | Architectural decisions          |
| `ROADMAP.md`           | Planned developments             |

---

# Development

Run the standard development checks:

```bash
ruff check .

python -m pytest
```

---

# Citation

If BayMorph contributes to published scientific work, please cite the corresponding software paper when available.

Until then, please cite the GitHub repository.

---

# Contributing

Contributions, bug reports, and feature requests are welcome.

Contribution guidelines will be provided in a future `CONTRIBUTING.md`.

---

# License

BayMorph is distributed under the MIT License.
