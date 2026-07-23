# BayMorph

<p align="center">
  <strong>Bayesian galaxy morphology made simple.</strong>
</p>

<p align="center">
  <a href="https://github.com/carlocannarozzophd/BayMorph/releases/tag/v0.2.0"><img alt="Version" src="https://img.shields.io/badge/version-v0.2.0-blue"></a>
  <img alt="Python" src="https://img.shields.io/badge/python-%E2%89%A53.11-green">
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-MIT-lightgrey"></a>
  <a href="https://github.com/carlocannarozzophd/BayMorph/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/carlocannarozzophd/BayMorph/actions/workflows/ci.yml/badge.svg"></a>
  <img alt="Tests" src="https://img.shields.io/badge/tests-283%20passing-brightgreen">
  <img alt="Status" src="https://img.shields.io/badge/status-active%20development-orange">
</p>

<p align="center">
  <a href="#installation">Installation</a> ·
  <a href="#quick-start">Quick start</a> ·
  <a href="notebooks/01_learn_baymorph.ipynb">Tutorial</a> ·
  <a href="docs/API.md">API</a> ·
  <a href="docs/VALIDATION.md">Validation</a> ·
  <a href="CONTRIBUTING.md">Contributing</a>
</p>

> [!IMPORTANT]
> BayMorph is under active scientific development. The current codebase provides a tested, configuration-driven framework for image preparation, Sérsic modelling, Bayesian inference, model comparison, and diagnostic generation. Validation across heterogeneous observational datasets remains ongoing.

Measuring galaxy structure is a fundamental task in observational astronomy. Quantities such as total flux, effective radius, Sérsic index, axis ratio, and position angle provide important constraints on galaxy formation and evolution. Recovering them reliably is nevertheless a genuine inference problem: the parameters are correlated, the likelihood can contain extended degeneracies, and several models may describe the same noisy image nearly equally well.

**BayMorph** is an open-source Python framework that approaches galaxy morphology from a Bayesian perspective. It combines reproducible image preparation, physically motivated Sérsic models, nested sampling, Bayesian evidence, and posterior diagnostics in a single configuration-driven workflow.

> **Galaxy morphology is fundamentally an inference problem, not merely an optimization problem.**

Instead of reporting only one best-fitting parameter vector, BayMorph characterizes the posterior distribution supported by the data. This makes parameter uncertainties, covariances, degeneracies, and model-comparison evidence explicit parts of the scientific result.

---

## Why BayMorph?

Many morphology workflows require separate tools or notebook-specific code for image preparation, target selection, masking, PSF construction, fitting, posterior analysis, and visualization. This can make the complete analysis difficult to reproduce and scientific assumptions difficult to audit.

BayMorph integrates these stages while keeping them modular. Scientific algorithms live in tested package modules; YAML files define complete analyses; notebooks remain lightweight interfaces rather than hidden implementations.

BayMorph is designed to help users:

- recover full posterior distributions rather than only point estimates;
- quantify correlations such as the Sérsic flux–radius–index degeneracy;
- compare competing models using Bayesian evidence;
- reproduce an analysis from a resolved configuration;
- inspect the exact prepared image, mask, noise map, PSF, and diagnostics used by the fit;
- extend individual workflow stages without redesigning the entire package.

---

## Key features

### Bayesian inference

- Nested sampling with **Dynesty 3.x**.
- Gaussian image likelihoods with optional normalization.
- Weighted posterior summaries and equal-weight posterior samples.
- Bayesian evidence and Bayes-factor model comparison.
- Configurable priors, convergence criteria, random seeds, and parallel execution.

### Galaxy models

- Single-Sérsic models.
- Double-Sérsic models.
- PSF-convolved rendering.
- Model-agnostic `render()` and `render_components()` interfaces.
- Shared result infrastructure for single- and multi-model workflows.

### Image preparation

- FITS loading and structured output management.
- Target selection from a seed position.
- Fixed and automatic cutout strategies.
- Image-weighted target-geometry estimation.
- Elliptical radial profiles and robust plateau detection.
- Target-specific sky-region construction.
- Empirical background and noise estimation.
- Connected-source masking and optional mask refinement.

### PSF handling

- Analytic Gaussian PSFs.
- Empirical PSF candidate selection.
- Candidate recentering, validation, stacking, persistence, and loading.
- PSF normalization checks before convolution and fitting.

### Diagnostics

- Posterior corner plots.
- Data, model, residual, and normalized-residual maps.
- Radial profiles.
- Posterior radial-profile bands.
- Image-preparation and reconstruction reports.
- PDF and image outputs suitable for scientific inspection.

### Software design

- Stable top-level public API.
- High-level `BayMorphRunner` orchestration.
- YAML-driven reproducibility.
- Packaged tutorial configuration and notebook.
- Command-line project initialization and example installation.
- Automated tests for scientific logic, workflows, packaging, CLI behavior, and resource consistency.

---

## Installation

BayMorph currently supports **Python 3.11 and later**. A dedicated Conda environment is recommended.

### Install from GitHub

```bash
conda create -n baymorph -c conda-forge python=3.11 -y
conda activate baymorph

python -m pip install --upgrade pip
python -m pip install \
  "git+https://github.com/carlocannarozzophd/BayMorph.git@main"
```

Verify the installation:

```bash
baymorph --version
baymorph --help
```

### Editable development install

```bash
git clone https://github.com/carlocannarozzophd/BayMorph.git
cd BayMorph

python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Run the test suite:

```bash
ruff check .
python -m pytest -q
```

> [!NOTE]
> A PyPI installation command will be added once the first public distribution is published. Until then, install BayMorph directly from the GitHub repository.

---

## Quick start

BayMorph analyses are defined through YAML configuration files and executed through `BayMorphRunner`.

```python
from baymorph import BayMorphRunner

runner = BayMorphRunner(
    "configs/learn_baymorph.yaml",
)

runner.prepare_image()
runner.fit()
runner.diagnostics()
```

The main products remain available on the runner:

```python
prepared = runner.image
fit_result = runner.results
profile_bands = runner.profile_bands
```

For workflows using an empirical PSF:

```python
runner.prepare_image()
runner.select_psf_candidates()
runner.build_psf()
runner.fit()
runner.diagnostics()
```

Each stage returns its products directly and stores them on the runner for later inspection.

> [!IMPORTANT]
> The `baymorph` command-line interface currently manages projects and packaged examples. Scientific workflows are executed through the Python API.

---

## Learn BayMorph

New users should begin with the official tutorial:

**[`notebooks/01_learn_baymorph.ipynb`](notebooks/01_learn_baymorph.ipynb)**

The notebook introduces a complete, fast, synthetic workflow:

1. inspect the official YAML configuration;
2. generate a reproducible Sérsic observation;
3. prepare the image and estimate the background and noise;
4. build the Gaussian PSF used by the model;
5. run a single-Sérsic Dynesty fit;
6. inspect posterior summaries and parameter degeneracies;
7. generate corner, residual, and radial-profile diagnostics;
8. compare recovered parameters with the injected truth without overinterpreting one noisy realization.

The mirrored configuration is available at:

```text
configs/learn_baymorph.yaml
baymorph/resources/configs/learn_baymorph.yaml
```

The mirrored notebook is available at:

```text
notebooks/01_learn_baymorph.ipynb
baymorph/resources/notebooks/01_learn_baymorph.ipynb
```

Repository and packaged resources are tested for byte-level consistency.

---

## Scientific workflow

A typical BayMorph analysis follows this sequence:

```text
FITS observation
       │
       ▼
Target identification and cutout
       │
       ▼
Background, noise, and masking
       │
       ▼
PSF loading or construction
       │
       ▼
Sérsic model and likelihood
       │
       ▼
Dynesty nested sampling
       │
       ▼
Posterior summaries and evidence
       │
       ▼
Corner, residual, and radial diagnostics
```

The configuration resolves all runtime choices before execution, including input paths, target position, cutout strategy, background and noise methods, masking, PSF type, models, priors, likelihood, sampler settings, model comparison, and diagnostics.

Resolved configurations and generated products are written to a structured run directory so that the complete analysis can be inspected after execution.

---

## Configuration

The official tutorial demonstrates the main configuration sections:

```yaml
object:
  name: learn_baymorph

data:
  image_path: runs/learn_baymorph/input/learn_baymorph.fits
  target_position:
    x: 80
    y: 80
  cutout:
    mode: fixed
    size: 161

outputs:
  root: runs
  overwrite: true

preparation:
  background:
    method: empirical
  noise:
    method: empirical

psf:
  type: gaussian

fit:
  models:
    - single_sersic
  likelihood:
    type: gaussian
  sampler:
    name: dynesty

diagnostics:
  corner: true
  residual: true
  radial_profile:
    binning: fixed
```

The complete validated configuration is available at [`configs/learn_baymorph.yaml`](configs/learn_baymorph.yaml).

---

## Public API

The stable top-level API currently exposes:

```python
from baymorph import (
    BayMorphRunner,
    CompositeModel,
    ConvolvedModel,
    DoubleSersicModel,
    EmpiricalPSF,
    GaussianPSF,
    ImageData,
    Observation,
    ObservationMetadata,
    PosteriorResult,
    RealFitProducts,
    SersicModel,
    SyntheticObservation,
    SyntheticSersic,
)
```

The complete API guide distinguishes stable, advanced, and internal interfaces:

**[`docs/API.md`](docs/API.md)**

---

## Architecture

BayMorph follows a layered architecture in which each package owns a focused responsibility:

```text
User code / notebook
        │
        ▼
BayMorphRunner
        │
        ▼
Pipeline orchestration
        │
        ├── image preparation
        ├── PSF workflows
        ├── fitting
        └── diagnostics
        │
        ▼
Scientific modules
        ├── core data abstractions
        ├── models and rendering
        ├── likelihoods and samplers
        ├── simulations
        ├── plotting
        └── result objects
```

This design separates scientific algorithms from orchestration and user interfaces. It allows low-level components to be tested independently while preserving a concise high-level workflow.

See **[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)** for the full design description and **[`DECISIONS.md`](DECISIONS.md)** for the rationale behind major architectural choices.

---

## Repository layout

```text
BayMorph/
├── baymorph/                  # Python package
│   ├── core/                  # observations and image abstractions
│   ├── fitting/               # likelihoods, priors, samplers, factories
│   ├── models/                # Sérsic and composite models
│   ├── pipeline/              # workflow orchestration
│   ├── plotting/              # diagnostic visualization
│   ├── resources/             # packaged YAML and notebook resources
│   ├── results/               # posterior and fit products
│   ├── simulation/            # synthetic sources, PSFs, and observations
│   └── runner.py              # high-level user interface
├── configs/                   # official YAML configurations
├── notebooks/                 # official tutorials
├── docs/                      # API, architecture, and validation guides
├── tests/                     # automated scientific and software tests
├── CHANGELOG.md
├── CITATION.cff
├── CONTRIBUTING.md
├── DECISIONS.md
├── ROADMAP.md
└── pyproject.toml
```

---

## Command-line interface

Copy the packaged tutorial resources:

```bash
baymorph examples baymorph_examples
```

Create a standalone project directory:

```bash
baymorph init MyBayMorphProject
```

The generated project contains:

```text
MyBayMorphProject/
├── configs/
├── notebooks/
├── data/
└── runs/
```

Use `--force` only when intentionally writing into an existing destination:

```bash
baymorph examples baymorph_examples --force
baymorph init MyBayMorphProject --force
```

---

## Diagnostics and posterior interpretation

BayMorph diagnostics are designed to expose the statistical structure of the fit rather than only its visual reconstruction.

### Corner plots

Corner plots use equal-weight posterior samples when available. They show marginal credible regions and correlations among parameters such as flux, effective radius, Sérsic index, axis ratio, orientation, centroid, and residual sky.

### Residual maps

Residual diagnostics compare the prepared data and posterior model on the same pixel grid. Normalized residuals use the same uncertainty map and mask supplied to the likelihood.

### Radial profiles

Radial diagnostics compare observed and model surface-brightness profiles and can include bands derived from equal-weight posterior realizations.

A visually excellent residual map does not imply that every parameter is independently well constrained. Sérsic flux, radius, and index can remain strongly degenerate even when the pixel-level fit is statistically acceptable. BayMorph therefore treats posterior structure as part of the result, not as a secondary visualization.

---

## Validation

BayMorph validation combines automated testing, controlled simulations, workflow regression tests, and targeted scientific experiments.

The test suite covers:

- configuration loading and resolution;
- FITS and observation abstractions;
- source detection, segmentation, and target selection;
- fixed and automatic cutouts;
- target geometry and radial profiles;
- background, sky-region, and noise estimation;
- masks and optional refinement;
- analytic and empirical PSFs;
- Sérsic, double-Sérsic, composite, and convolved models;
- likelihoods, priors, Dynesty integration, and sampler factories;
- posterior summaries and weighted statistics;
- residual, corner, and radial-profile diagnostics;
- high-level runner workflows;
- packaged resources, CLI behavior, builds, and public imports.

Dedicated preparation experiments over multiple noise seeds and subpixel centers showed subpixel centroid recovery and small background errors for the official synthetic setup. Controlled fits also demonstrated that shifts in flux, effective radius, and Sérsic index follow the expected posterior degeneracy rather than an obvious simulation–renderer inconsistency.

An external pySersic cross-check recovered broadly consistent source geometry, centroid, residual sky, flux, and effective radius, while showing renderer-dependent differences in Sérsic index and fit quality. This comparison is treated as a qualitative independent check, not as proof that distinct numerical renderers must return identical parameters.

See **[`docs/VALIDATION.md`](docs/VALIDATION.md)** for the validation framework and current limitations.

---

## Current scope and limitations

BayMorph is a research software project under active development. Current users should keep the following scope in mind:

- The public tutorial currently covers a single synthetic single-Sérsic workflow.
- Single- and double-Sérsic infrastructure exists, but broad real-galaxy validation is still expanding.
- The standard background model is currently constant after empirical preparation; spatially varying backgrounds remain future work.
- The primary sampling backend is Dynesty.
- Sérsic parameters can be strongly correlated, and one noisy realization should not be expected to place every injected truth inside a 68% credible interval.
- Results from different fitting packages may differ because of pixel integration, Sérsic normalization, PSF convolution, renderer approximations, likelihood details, priors, and point-estimate versus posterior summaries.

Scientific conclusions should therefore be based on posterior distributions, residual diagnostics, model checks, and validation appropriate to the dataset being analyzed.

---

## Documentation

- [Project overview](README.md)
- [Learn BayMorph notebook](notebooks/01_learn_baymorph.ipynb)
- [Documentation index](docs/README.md)
- [Public API guide](docs/API.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Validation](docs/VALIDATION.md)
- [Architectural decisions](DECISIONS.md)
- [Roadmap](ROADMAP.md)
- [Changelog](CHANGELOG.md)
- [Contribution guide](CONTRIBUTING.md)

---

## Contributing

Contributions, bug reports, validation datasets, documentation improvements, and scientific discussions are welcome.

Before submitting a pull request, install the development dependencies and run:

```bash
ruff check .
python -m pytest -q
python -m build
python -m twine check dist/*
```

Please read **[`CONTRIBUTING.md`](CONTRIBUTING.md)** for development conventions, testing expectations, and pull-request guidance.

---

## Citation

If BayMorph contributes to your research, please cite the software using the metadata in **[`CITATION.cff`](CITATION.cff)**.

GitHub can export this metadata directly through the repository's **Cite this repository** interface.

Current citation metadata:

```text
Carlo Cannarozzo. BayMorph: Bayesian morphological modelling of
astronomical images. Version 0.2.0 (2026).
```

Citation information will be updated when a new tagged release or archival DOI becomes available.

---

## Roadmap

Current development priorities include:

- validation on heterogeneous real observations;
- broader single- and double-Sérsic recovery experiments;
- validation of automatic cutouts, backgrounds, noise maps, and empirical PSFs;
- improved evidence and model-comparison summaries;
- synchronized release documentation and reproducible tutorial products.

Longer-term directions include:

- non-constant background models;
- correlated-noise likelihoods;
- additional morphology components;
- alternative sampling backends;
- spatially varying PSFs;
- multi-band joint fitting;
- survey-scale processing and catalogue production.

See **[`ROADMAP.md`](ROADMAP.md)** for the complete development plan.

---

## License

BayMorph is distributed under the **MIT License**. See **[`LICENSE`](LICENSE)** for the full text.

---

## Acknowledgements

BayMorph is built on the scientific Python ecosystem, including NumPy, SciPy, Astropy, Photutils, Matplotlib, PyYAML, and Dynesty.

The project is developed with an emphasis on scientific transparency, reproducibility, modular software design, and rigorous Bayesian interpretation of galaxy morphology.
