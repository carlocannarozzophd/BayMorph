# BayMorph

BayMorph is an open-source Python package for Bayesian morphological modelling of astronomical images.

## Status

Early development.

Current target: single Sérsic fitting on FITS images using Bayesian nested sampling.

## Planned Features

- FITS image input
- Background and noise estimation
- Source detection and masking
- PSF handling
- Sérsic profile modelling
- Nested sampling with Bayesian evidence
- Model comparison
- Notebook-based workflow

## Installation

```bash
pip install -e .
```

## Development

```bash
pip install -e ".[dev]"
pytest
```

## Current Development Roadmap

Current MVP:

```text
Synthetic Sérsic galaxy
        ↓
PSF convolution
        ↓
Noise generation
        ↓
Synthetic FITS creation
        ↓
Bayesian fitting
        ↓
Parameter recovery
```

## License

MIT License.