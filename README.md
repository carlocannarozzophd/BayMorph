# BayMorph

> **Status:** Active development. Core functionality has been scientifically validated on synthetic datasets and is currently undergoing validation on real galaxies.

BayMorph is a Python library for **Bayesian structural decomposition of galaxies**. It provides an end-to-end workflow from astronomical images to Bayesian parameter inference and model comparison, integrating image preparation, source detection and masking, PSF handling, physically motivated galaxy models, and publication-quality diagnostics within a unified framework.

The project is designed to make Bayesian galaxy fitting reproducible, configurable, and easy to apply to both synthetic and real astronomical observations.

---

# Workflow

```text
FITS image
    ↓
Image preparation
    ↓
Background estimation
    ↓
Noise estimation
    ↓
Source detection & masking
    ↓
PSF construction
    ↓
Bayesian galaxy fitting
    ↓
Model comparison
    ↓
Posterior analysis & diagnostics
```

---

# Features

## Image Preparation

- FITS image loading
- Automatic target cutout extraction
- Background estimation and subtraction
- Noise-map generation
- Source detection and segmentation
- Automatic contaminant masking
- Configuration-driven workflows

## Galaxy Modelling

- Single-Sérsic models
- Double-Sérsic models
- PSF-convolved forward modelling
- Flexible parameter definitions and priors

## Bayesian Inference

- Dynesty nested sampling
- Bayesian evidence estimation
- Bayesian model comparison
- Parallel sampling
- Reproducible analyses through configurable random seeds

## Diagnostics

- Publication-quality model/residual figures
- Corner plots
- Radial-profile diagnostics
- Posterior profile bands
- Fit summary reports

---

# Installation

Clone the repository:

```bash
git clone <repository-url>
cd BayMorph
```

Create a Python environment:

```bash
conda create -n baymorph python=3.11
conda activate baymorph
```

Install BayMorph:

```bash
pip install -e .
```

Install development dependencies:

```bash
pip install -e ".[dev]"
```

---

# Examples

The repository contains three official notebooks demonstrating the standard BayMorph workflow.

| Notebook | Description |
|----------|-------------|
| `01_SingleSersic_Validation.ipynb` | Validate Bayesian recovery of a synthetic single-Sérsic galaxy with known truth parameters. |
| `02_MockPipeline_Validation.ipynb` | Demonstrate the complete mock-observation workflow, including PSF convolution, sky background, Gaussian noise, image preparation, Bayesian fitting, and diagnostics. |
| `TARGET_BayMorph_Test.ipynb` | Apply BayMorph to a real galaxy using the standard analysis pipeline. |

The corresponding YAML configuration files are provided in the `examples/` directory.

---

# Scientific Validation

The mathematical core of BayMorph has been incrementally validated using synthetic galaxies with known ground-truth parameters.

Completed validation includes:

- Single-Sérsic parameter recovery
- PSF-convolved forward modelling
- Gaussian-noise likelihood validation
- Complete mock-observation pipeline
- End-to-end pipeline validation

Current scientific development focuses on:

- Empirical PSF validation
- Contaminant masking improvements
- Local background estimation
- Validation on larger samples of real galaxies

The current real-galaxy validation target is **TARGET**, which serves as the primary end-to-end scientific demonstration of the pipeline.

---

# Testing

The automated test suite currently validates:

- FITS I/O
- Background estimation
- Noise handling
- Source detection and masking
- PSF construction
- Sérsic models
- Bayesian inference
- Model comparison
- Pipeline configuration
- Real-image workflow components

---

# Documentation

Additional documentation is available in:

- `docs/README.md`
- `docs/ARCHITECTURE.md`
- `ROADMAP.md`

---

# License

Released under the **MIT License**.