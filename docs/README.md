# BayMorph Documentation

BayMorph is a Bayesian framework for the morphological analysis of astronomical images.

The project combines image preparation, PSF handling, galaxy modelling, Bayesian inference, model comparison, and scientific diagnostics within a unified workflow.

## Current Capabilities

### Image Preparation

- FITS image ingestion
- Automatic target cutout extraction
- Background estimation and subtraction
- Source-masked noise-map generation
- Source detection and segmentation
- Automatic contaminant masking

### Point Spread Functions

- Gaussian PSF models
- Empirical PSF construction from field stars
- PSF candidate selection and filtering
- PSF-convolved galaxy modelling

### Galaxy Modelling

- Single-Sérsic models
- Double-Sérsic models
- Bulge + disk style decompositions
- PSF-convolved forward modelling

### Bayesian Inference

- Generic Dynesty fitting
- Bayesian evidence estimation
- Evidence uncertainty estimation (`logZerr`)
- Bayes-factor model comparison
- Reproducible sampling through random seeds
- Parallel nested sampling
- Configurable convergence thresholds (`dlogz`)
- Multi-model fitting workflows

### Posterior Analysis

- Weighted posterior statistics
- Equal-weight posterior samples
- Posterior summaries
- Corner plots
- Posterior profile bands

### Diagnostics

- Model-image generation
- Residual-image generation
- Sigma-normalized residual maps
- Radial-profile diagnostics
- Radial residual diagnostics
- Fit summaries
- Bayesian model-comparison summaries

---

## Current Workflow

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

## Documentation Structure

### Architecture

See:

```text
docs/ARCHITECTURE.md
```

for package structure, workflow design, implementation details, and architectural decisions.

### Roadmap

See:

```text
ROADMAP.md
```

for current development priorities, validation status, and planned future developments.

---

## Development Status

BayMorph is currently in an active scientific validation phase.

The software has successfully completed its first end-to-end real-galaxy validation campaign, including:

- image preparation
- source detection and segmentation
- contaminant masking
- Gaussian and empirical PSF workflows
- single-Sérsic fitting
- double-Sérsic fitting
- Bayesian model comparison
- residual diagnostics
- posterior diagnostics

Current validation target:

```text
TARGET
```

Interfaces and workflow details may continue to evolve before the first public release, but the core scientific workflow is operational and under active validation on real astronomical datasets.