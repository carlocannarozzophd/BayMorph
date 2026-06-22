# BayMorph

BayMorph is a Bayesian framework for the morphological analysis of astronomical images.

The project is designed to provide a unified workflow from image preparation to Bayesian model comparison, combining physically motivated galaxy models, realistic observational effects, and posterior-based parameter inference.

BayMorph currently supports:

- FITS-based image preparation
- Background estimation and subtraction
- Noise-map generation
- Source detection and segmentation
- Empirical and Gaussian PSF workflows
- Single-Sérsic fitting
- Double-Sérsic fitting
- Bayesian model comparison using nested sampling and Bayes factors

The software is currently undergoing scientific validation on real astronomical datasets. Core functionality is operational and actively tested, but interfaces, diagnostics, and workflow details may continue to evolve before the first public release.

---

# Current Workflow

```text
FITS image
    ↓
Image preparation
    ↓
Source detection and masking
    ↓
PSF construction
    ↓
Single-Sérsic fit
Double-Sérsic fit
    ↓
Bayesian model comparison
    ↓
Diagnostics
```

---

# Features

## Image Preparation

- FITS image loading and metadata handling
- Automatic target cutout extraction
- Background estimation and subtraction
- Empirical noise-map generation
- Source detection and segmentation
- Automatic contaminant masking
- Configuration-driven workflows

## Point Spread Function (PSF)

- Gaussian PSF models
- Empirical PSF construction from field stars
- PSF candidate selection and quality filtering
- PSF-convolved galaxy models

## Galaxy Modelling

- Single-Sérsic models
- Double-Sérsic models
- Bulge + disk style decompositions
- PSF-convolved forward modelling

## Bayesian Inference

- Generic Dynesty nested-sampling framework
- Flexible parameter definitions through `Parameter` and `ParameterSet`
- Support for masked likelihood evaluation
- Bayesian evidence estimation
- Bayes-factor model comparison

## Posterior Analysis

- Posterior sample storage
- Weighted posterior statistics
- Best-fit parameter estimates
- Median posterior estimates
- Dynesty posterior reconstruction

## Diagnostics

- Model-image generation
- Residual-image generation
- Normalized residual maps
- Fit summaries and metadata products

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

Install BayMorph:

```bash
pip install -e .
```

Install development dependencies:

```bash
pip install -e ".[dev]"
```

---

# Example Configuration

```yaml
object:
  name: TARGET

data:
  image_path: path/to/image.fits

outputs:
  root: outputs

fit:
  models:
    - single_sersic
    - double_sersic

  sampler:
    name: dynesty
    nlive: 300

  comparison:
    enabled: true
    metric: bayes_factor
```

---

# Current Status

BayMorph has successfully completed its first end-to-end real-galaxy tests, including:

- Image preparation
- Source detection and segmentation
- Empirical and Gaussian PSF workflows
- Single-Sérsic fitting
- Double-Sérsic fitting
- Bayesian model comparison

The software is currently in an active validation phase.

Current development priorities include:

- Improved empirical PSF construction
- Automatic diagnostic generation
- Posterior-profile diagnostics
- Interactive masking workflows
- Expanded model libraries

---

# Validation

The automated test suite currently validates:

- FITS I/O
- Background estimation
- Noise handling
- Source detection and masking
- PSF construction
- Sérsic models
- Dynesty fitting
- Bayesian model comparison
- Real-image workflow components

---

# Documentation

Additional documentation is available in:

- `docs/README.md`
- `docs/ARCHITECTURE.md`
- `ROADMAP.md`

---

# License

MIT License.