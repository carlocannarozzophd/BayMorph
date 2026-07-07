# BayMorph

```{=html}
<p align="center">
```
`<strong>`{=html}Bayesian galaxy morphology from image to
evidence`</strong>`{=html}
```{=html}
</p>
```
```{=html}
<p align="center">
```
`<img alt="version" src="https://img.shields.io/badge/version-v0.2.0-blue">`{=html}
`<img alt="python" src="https://img.shields.io/badge/python-3.11%2B-green">`{=html}
`<img alt="tests" src="https://img.shields.io/badge/tests-159%2F159%20passed-brightgreen">`{=html}
`<img alt="license" src="https://img.shields.io/badge/license-MIT-lightgrey">`{=html}
`<img alt="status" src="https://img.shields.io/badge/status-active%20development-orange">`{=html}
```{=html}
</p>
```
**BayMorph** is an open-source Python framework for **Bayesian
structural decomposition of galaxies** from astronomical imaging.

It takes a galaxy image through a complete, reproducible workflow: image
preparation, background and noise estimation, masking, PSF handling,
Sérsic modelling, Bayesian inference, model comparison, and diagnostic
plots.

> \[!IMPORTANT\] BayMorph is currently at **v0.2.0**. The public
> workflow, notebooks, documentation, and automated tests are in place.
> Scientific validation on larger real-galaxy samples is ongoing.

------------------------------------------------------------------------

## Why BayMorph?

Whether you are validating a new morphological model, analysing a single
galaxy, or processing an entire sample, BayMorph provides a single,
reproducible workflow.

BayMorph is designed for cases where galaxy fitting is not just a single
optimization call, but a full scientific workflow.

  -----------------------------------------------------------------------
  Need                                Use BayMorph to
  ----------------------------------- -----------------------------------
  Too many manual steps from image to Run a complete workflow from FITS
  final fit?                          image to diagnostics with one
                                      high-level runner.

  Need a controlled mock galaxy?      Generate and fit simulated Sérsic
                                      galaxies with known input
                                      parameters.

  Working with real observations?     Prepare real FITS images, estimate
                                      background/noise, build or load a
                                      PSF, and fit galaxy models.

  Need to compare models?             Use Bayesian evidence and Bayes
                                      factors to compare single- and
                                      double-Sérsic models.

  Want reproducible analyses?         Control the workflow with YAML
                                      configuration templates and store
                                      resolved run products.

  Need publication-quality            Produce residual maps, corner
  diagnostics?                        plots, radial profiles, posterior
                                      summaries, and fit reports.
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## Workflow-first galaxy modelling

BayMorph treats galaxy structural decomposition as a complete scientific
workflow rather than an isolated fitting problem.

``` text
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
Bayesian model comparison
    ↓
Posterior analysis and diagnostics
```

The goal is to make galaxy structural analysis transparent,
reproducible, inspectable, and easy to apply to both simulated and real
observations.

------------------------------------------------------------------------

## Main features

  -----------------------------------------------------------------------
  Area                                Current support
  ----------------------------------- -----------------------------------
  **Image preparation**               FITS loading, target cutouts,
                                      background subtraction, noise maps,
                                      source detection, segmentation,
                                      contaminant masking

  **PSF handling**                    Gaussian PSFs, empirical PSFs from
                                      field stars, user-provided PSFs,
                                      PSF-convolved modelling

  **Galaxy models**                   Single-Sérsic and double-Sérsic
                                      models with configurable priors and
                                      parameter bounds

  **Bayesian inference**              `dynesty` nested sampling,
                                      posterior samples, Bayesian
                                      evidence, evidence uncertainties,
                                      Bayes factors

  **Diagnostics**                     Residual maps, sigma-normalized
                                      residuals, corner plots, radial
                                      profiles, fit reports,
                                      model-comparison products

  **Workflow control**                YAML-driven analyses through the
                                      high-level `BayMorphRunner` API
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## Installation

Clone the repository:

``` bash
git clone https://github.com/carlocannarozzophd/BayMorph.git
cd BayMorph
```

Create and activate a Python environment:

``` bash
conda create -n baymorph python=3.11
conda activate baymorph
```

Install BayMorph:

``` bash
pip install -e .
```

For development:

``` bash
pip install -e ".[dev]"
```

------------------------------------------------------------------------

## Quick start

BayMorph is intended to be used through the high-level workflow API.

``` python
import matplotlib.pyplot as plt

from baymorph import BayMorphRunner, load_resolved_config

config = load_resolved_config("configs/simulated_galaxy.yaml")

runner = BayMorphRunner(config)
results = runner.run()

plt.show()
```

`BayMorphRunner` orchestrates the complete workflow. Fitting products
are exposed through high-level objects such as `RealFitProducts` and
`PosteriorResult`.

------------------------------------------------------------------------

## Configuration templates

Standard configuration templates are stored in [`configs/`](configs/):

``` text
configs/
├── simulated_galaxy.yaml
└── real_galaxy.yaml
```

The real-galaxy template is intentionally generic. Notebooks update only
the target name and FITS image path:

``` yaml
object:
  name: galaxy

data:
  image_path: path/to/image.fits
```

All other workflow choices are controlled by the YAML configuration.

------------------------------------------------------------------------

## Official notebooks

BayMorph includes three public notebooks.

  --------------------------------------------------------------------------------------------------------------------------
  Notebook                                                                               Purpose
  -------------------------------------------------------------------------------------- -----------------------------------
  [`01_Simulated_Galaxy_Analysis.ipynb`](notebooks/01_Simulated_Galaxy_Analysis.ipynb)   Demonstrates the complete workflow
                                                                                         on a simulated galaxy with known
                                                                                         input parameters.

  [`02_Real_Galaxy_Analysis.ipynb`](notebooks/02_Real_Galaxy_Analysis.ipynb)             Demonstrates the workflow on a
                                                                                         single real galaxy using the
                                                                                         generic configuration template.

  [`03_Multi_Galaxy_Analysis.ipynb`](notebooks/03_Multi_Galaxy_Analysis.ipynb)           Automatically discovers and
                                                                                         processes all FITS images in a
                                                                                         directory using the same workflow.
  --------------------------------------------------------------------------------------------------------------------------

Recommended order:

1.  [`01_Simulated_Galaxy_Analysis.ipynb`](notebooks/01_Simulated_Galaxy_Analysis.ipynb)
2.  [`02_Real_Galaxy_Analysis.ipynb`](notebooks/02_Real_Galaxy_Analysis.ipynb)
3.  [`03_Multi_Galaxy_Analysis.ipynb`](notebooks/03_Multi_Galaxy_Analysis.ipynb)

------------------------------------------------------------------------

## Repository structure

``` text
baymorph/     Python package source code
configs/      YAML configuration templates
notebooks/    tutorial and validation notebooks
docs/         technical documentation
tests/        automated test suite
runs/         local pipeline outputs
```

Pipeline outputs are written to `runs/`:

``` text
runs/
└── object_name/
    ├── config_resolved.yaml
    ├── input/
    ├── image_preparation/
    ├── psf/
    ├── fits/
    └── model_comparison/
```

The `runs/` directory is intended for local analyses and is ignored by
Git except for `.gitkeep`.

------------------------------------------------------------------------

## Validation status

BayMorph v0.2.0 has been validated through:

-   synthetic single-Sérsic recovery tests;
-   PSF-convolved forward-modelling tests;
-   Gaussian-noise likelihood tests;
-   complete simulated-galaxy workflow tests;
-   real-galaxy workflow demonstrations;
-   automated unit and integration tests.

Run the test suite with:

``` bash
pytest
```

Current status:

``` text
159 / 159 tests passed
```

------------------------------------------------------------------------

## Learn more

### Tutorials

-   [`01_Simulated_Galaxy_Analysis.ipynb`](notebooks/01_Simulated_Galaxy_Analysis.ipynb)
-   [`02_Real_Galaxy_Analysis.ipynb`](notebooks/02_Real_Galaxy_Analysis.ipynb)
-   [`03_Multi_Galaxy_Analysis.ipynb`](notebooks/03_Multi_Galaxy_Analysis.ipynb)

### Documentation

  ------------------------------------------------------------------------------------
  Document                                         Purpose
  ------------------------------------------------ -----------------------------------
  [`docs/README.md`](docs/README.md)               Documentation index and recommended
                                                   reading order

  [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)   Internal architecture, public API,
                                                   and workflow organization

  [`docs/VALIDATION.md`](docs/VALIDATION.md)       Validation strategy, current
                                                   validation status, and limitations

  [`ROADMAP.md`](ROADMAP.md)                       Near-term, medium-term, and
                                                   long-term development roadmap

  [`DECISIONS.md`](DECISIONS.md)                   Accepted architectural decisions
                                                   for BayMorph

  [`README.md`](README.md)                         Repository homepage and quick-start
                                                   guide
  ------------------------------------------------------------------------------------

------------------------------------------------------------------------

## Current development status

BayMorph currently supports:

-   single-band analyses;
-   single-Sérsic and double-Sérsic models;
-   Bayesian model comparison;
-   simulated-galaxy workflows;
-   real-galaxy workflows;
-   multi-galaxy batch processing through notebooks.

Development is focused on larger real-galaxy validation samples, more
robust empirical PSF construction, additional galaxy models, and
release-quality documentation.

------------------------------------------------------------------------

## License

BayMorph is released under the MIT License.
