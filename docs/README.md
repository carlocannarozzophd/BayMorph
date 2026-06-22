# BayMorph Documentation

BayMorph is a Bayesian framework for the morphological analysis of astronomical images.

The project combines image preparation, PSF handling, galaxy modelling, Bayesian inference, and model comparison within a unified workflow.

## Current Capabilities

### Image Preparation

- FITS image ingestion
- Automatic target cutout extraction
- Background estimation and subtraction
- Noise-map generation
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

### Bayesian Inference

- Generic Dynesty fitting
- Bayesian evidence estimation
- Bayes-factor model comparison
- Multi-model fitting workflows

---

## Documentation Structure

### Architecture

See:

```text
docs/ARCHITECTURE.md
```

for package structure, workflow design, and implementation details.

### Roadmap

See:

```text
ROADMAP.md
```

for current development priorities and planned features.

---

## Development Status

BayMorph is currently in an active scientific validation phase.

The software has successfully completed its first end-to-end real-galaxy tests, including:

- image preparation
- source detection
- PSF construction
- single-Sérsic fitting
- double-Sérsic fitting
- Bayesian model comparison

Interfaces and workflow details may continue to evolve before the first public release.