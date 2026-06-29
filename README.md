# BayMorph

> **Status:** active development. BayMorph has been validated on synthetic datasets and includes an end-to-end real-galaxy demonstration, but interfaces may still evolve before a stable public release.

BayMorph is an open-source Python package for **Bayesian morphological modelling of astronomical images**. It provides a configurable workflow from FITS images to structural galaxy fitting, Bayesian model comparison, and publication-quality diagnostics.

BayMorph is designed to be more than a fitting routine: it is a reproducible scientific workflow for galaxy structural decomposition, including image preparation, background and noise estimation, masking, PSF handling, Bayesian inference, and diagnostic reporting.

> **BayMorph is not designed as a standalone fitting engine only.**
>
> Many morphological-analysis tools in the literature focus primarily on the optimization or inference step, assuming that the image, mask, PSF, noise model, and diagnostic workflow have already been prepared externally.
>
> **BayMorph instead adopts a pipeline-first philosophy:** it treats galaxy structural decomposition as a complete, reproducible scientific workflow, from FITS image preparation to Bayesian model comparison and diagnostic reporting.
>
> In this sense, BayMorph is intended to help users not only fit a galaxy model, but also document, reproduce, inspect, and scientifically validate the full analysis chain.

---

## Overview

BayMorph combines:

- FITS image ingestion and target cutout extraction
- background estimation and subtraction
- source detection, segmentation, and contaminant masking
- Gaussian or empirical PSF handling
- single-Sérsic and double-Sérsic galaxy models
- PSF-convolved forward modelling
- Bayesian inference with `dynesty`
- Bayesian evidence and model comparison
- posterior, residual, radial-profile, and report diagnostics

The standard workflow is:

```text
FITS image
    ↓
Image preparation
    ↓
Background and noise estimation
    ↓
Source detection and masking
    ↓
PSF construction or PSF loading
    ↓
Bayesian galaxy fitting
    ↓
Model comparison
    ↓
Posterior analysis and diagnostics
```

---

## Features

### Image preparation

- FITS image loading
- target cutout generation
- background estimation and subtraction
- source-masked noise-map estimation
- source detection and segmentation
- contaminant masking
- configuration-driven workflows

### PSF handling

- analytic Gaussian PSFs
- empirical PSF construction from field stars
- user-provided PSFs
- PSF-convolved galaxy modelling

### Galaxy modelling

- single-Sérsic models
- double-Sérsic models
- configurable priors and parameter bounds
- model registry designed for future model extensions

### Bayesian inference

- nested sampling with `dynesty`
- posterior sampling
- Bayesian evidence estimation
- evidence uncertainty storage
- Bayesian model comparison
- reproducible sampling through random seeds
- parallel likelihood evaluation

### Diagnostics

- mock-observation and reconstruction reports
- data/model/residual figures
- sigma-normalized residual maps
- corner plots
- radial-profile diagnostics
- fit-summary products
- Bayesian model-comparison products

---

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd BayMorph
```

Create and activate a Python environment:

```bash
conda create -n baymorph python=3.11
conda activate baymorph
```

Install BayMorph:

```bash
pip install -e .
```

For development:

```bash
pip install -e ".[dev]"
```

---

## Quick start

A minimal mock-pipeline workflow is:

```python
import matplotlib.pyplot as plt

from baymorph import (
    load_resolved_config,
    make_observation_from_config,
    run_image_preparation,
    run_real_fit,
)

config = load_resolved_config("configs/mock_pipeline.yaml")

observation = make_observation_from_config(config)
observation.to_fits(config["data"]["image_path"])

preparation = run_image_preparation(config)
fit = run_real_fit(config)

fit.report(observation)
plt.show()
```

For a single-model configuration, `run_real_fit(config)` returns a `RealFitProducts` object. For a multi-model configuration, it returns a dictionary keyed by model name.

---

## Repository structure

```text
baymorph/     Python package source code
configs/      YAML configuration files for standard runs
notebooks/    tutorial and validation notebooks
runs/         local pipeline outputs, ignored by git except .gitkeep
tests/        automated test suite
docs/         technical documentation
```

Pipeline products are written to `runs/`. Each run has a structured layout:

```text
runs/
└── object_name/
    ├── config_resolved.yaml
    ├── input/
    ├── image_preparation/
    ├── psf/
    ├── fits/
    └── model_comparison/
```

---

## Configuration files

Standard configuration files are stored in `configs/`:

```text
configs/
├── mock_pipeline.yaml
├── real_pipeline.yaml
└── target.yaml
```

These files define input data, output locations, image-preparation settings, masking options, PSF configuration, model choices, priors, sampler settings, and diagnostics.

---

## Tutorials and validation notebooks

BayMorph includes three main notebooks:

| Notebook | Purpose |
|---|---|
| `01_SingleSersic_Validation.ipynb` | Validates the Bayesian fitter on a controlled synthetic single-Sérsic galaxy with known truth parameters. |
| `02_MockPipeline_Validation.ipynb` | Validates the complete end-to-end mock pipeline, from YAML configuration to FITS generation, image preparation, fitting, and diagnostics. |
| `TARGET_BayMorph_Test.ipynb` | Demonstrates the standard workflow on a real galaxy. |

The notebooks are intended to be read in sequence:

1. **Single-Sérsic validation** checks that the fitting machinery can recover known input parameters.
2. **Mock pipeline validation** checks that the complete configuration-driven workflow runs from FITS input to diagnostic report.
3. **TARGET** demonstrates the workflow on observational data.

---

## Scientific validation

Current validation includes:

- single-Sérsic recovery on synthetic data
- PSF-convolved forward modelling tests
- Gaussian-noise likelihood tests
- complete mock-observation pipeline tests
- end-to-end workflow tests
- real-galaxy demonstration on TARGET

The automated test suite contains more than 160 tests covering modelling, fitting, plotting, image preparation, PSF handling, configuration, and pipeline workflows.

Run the tests with:

```bash
pytest
```

---

## Documentation

Additional documentation is available in:

- `docs/README.md`
- `docs/ARCHITECTURE.md`
- `docs/VALIDATION.md`
- `ROADMAP.md`
- `DECISIONS.md`

---

## License

BayMorph is released under the MIT License.
