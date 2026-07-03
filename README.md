# BayMorph

```{=html}
<p align="center">
```
`<strong>`{=html}Bayesian structural decomposition of galaxies from
astronomical images`</strong>`{=html}
```{=html}
</p>
```
```{=html}
<p align="center">
```
`<img alt="version" src="https://img.shields.io/badge/version-v0.1.0-blue">`{=html}
`<img alt="python" src="https://img.shields.io/badge/python-3.11%2B-green">`{=html}
`<img alt="tests" src="https://img.shields.io/badge/tests-159%2F159%20passed-brightgreen">`{=html}
`<img alt="license" src="https://img.shields.io/badge/license-MIT-lightgrey">`{=html}
`<img alt="status" src="https://img.shields.io/badge/status-active%20development-orange">`{=html}
```{=html}
</p>
```
BayMorph is an open-source Python framework for **Bayesian galaxy
morphology**. It provides a complete, reproducible workflow from
astronomical FITS images to structural modelling, Bayesian model
comparison, and publication-quality diagnostics.

> \[!IMPORTANT\] **Status:** BayMorph **v0.1.0**. The public workflow,
> documentation, notebooks, and automated test suite are available.
> Scientific validation on larger real-galaxy samples is ongoing.

------------------------------------------------------------------------

# Why BayMorph?

BayMorph is designed for astronomers who want a **complete workflow**,
not just a galaxy fitting routine.

  -----------------------------------------------------------------------
  Want to...                          BayMorph helps you...
  ----------------------------------- -----------------------------------
  Fit a galaxy from a FITS image      Run the complete workflow from
                                      image preparation to Bayesian
                                      inference.

  Generate mock galaxies              Create simulated galaxies and
                                      observations with known input
                                      parameters.

  Analyse real observations           Prepare FITS images, estimate
                                      background and noise, construct or
                                      load a PSF, and fit galaxy models.

  Compare different models            Use Bayesian evidence and Bayes
                                      factors to compare competing
                                      morphological models.

  Process many galaxies               Analyse complete directories of
                                      FITS images using the same
                                      configuration template.

  Produce publication-quality figures Automatically generate residual
                                      maps, corner plots, radial
                                      profiles, and summary reports.
  -----------------------------------------------------------------------

------------------------------------------------------------------------

# Workflow-first philosophy

BayMorph treats galaxy structural decomposition as a complete scientific
workflow.

``` text
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

The scientific workflow is driven by configuration files and exposed
through the high-level `BayMorphRunner` interface.

------------------------------------------------------------------------

# Installation

``` bash
git clone https://github.com/carlocannarozzophd/BayMorph.git
cd BayMorph

conda create -n baymorph python=3.11
conda activate baymorph

pip install -e .
```

Development installation:

``` bash
pip install -e ".[dev]"
```

------------------------------------------------------------------------

# Quick start

``` python
from baymorph import BayMorphRunner, load_resolved_config

config = load_resolved_config("configs/simulated_galaxy.yaml")

runner = BayMorphRunner(config)
results = runner.run()
```

------------------------------------------------------------------------

# Official notebooks

  --------------------------------------------------------------------------------------------------------------------------
  Notebook                                                                               Purpose
  -------------------------------------------------------------------------------------- -----------------------------------
  [`01_Simulated_Galaxy_Analysis.ipynb`](notebooks/01_Simulated_Galaxy_Analysis.ipynb)   Complete workflow on a simulated
                                                                                         galaxy with known input parameters.

  [`02_Real_Galaxy_Analysis.ipynb`](notebooks/02_Real_Galaxy_Analysis.ipynb)             Complete workflow on a real galaxy
                                                                                         using the generic configuration
                                                                                         template.

  [`03_Multi_Galaxy_Analysis.ipynb`](notebooks/03_Multi_Galaxy_Analysis.ipynb)           Automatically processes all FITS
                                                                                         images in a directory.
  --------------------------------------------------------------------------------------------------------------------------

------------------------------------------------------------------------

# Documentation

## Getting started

-   [`01_Simulated_Galaxy_Analysis.ipynb`](notebooks/01_Simulated_Galaxy_Analysis.ipynb)
-   [`02_Real_Galaxy_Analysis.ipynb`](notebooks/02_Real_Galaxy_Analysis.ipynb)
-   [`03_Multi_Galaxy_Analysis.ipynb`](notebooks/03_Multi_Galaxy_Analysis.ipynb)

## Technical documentation

-   [`docs/README.md`](docs/README.md)
-   [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
-   [`docs/VALIDATION.md`](docs/VALIDATION.md)

## Project documents

-   [`DECISIONS.md`](DECISIONS.md)
-   [`ROADMAP.md`](ROADMAP.md)

------------------------------------------------------------------------

# Current status

-   Version **0.1.0**
-   **159 / 159** automated tests passing
-   Active development
-   MIT License

------------------------------------------------------------------------

# License

BayMorph is released under the MIT License.
