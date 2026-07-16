# BayMorph

<p align="center">
  <strong>Bayesian structural modelling of galaxies from astronomical images</strong>
</p>

<p align="center">
  <img alt="Version" src="https://img.shields.io/badge/version-v0.2.0-blue">
  <img alt="Python" src="https://img.shields.io/badge/python-%E2%89%A53.10-green">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-lightgrey">
  <img alt="Status" src="https://img.shields.io/badge/status-active%20development-orange">
</p>

> [!IMPORTANT]
> BayMorph is under active scientific development. Version 0.2.0 provides an end-to-end, configuration-driven workflow for image preparation, Sérsic-based Bayesian fitting, model comparison, and diagnostics. Validation on heterogeneous observational samples is ongoing.

BayMorph is an open-source Python package for **Bayesian morphological analysis of galaxies**. It is designed as a workflow-oriented scientific library: notebooks remain lightweight, while image preparation, fitting, diagnostics, and output management live in reusable package code.

## Why BayMorph?

BayMorph combines tasks that are often handled by several independent tools:

- FITS image preparation;
- target identification and segmentation;
- adaptive cutout construction;
- source masking;
- background and noise estimation;
- PSF loading or construction;
- Sérsic and composite-model rendering;
- Bayesian nested sampling with Dynesty;
- posterior summaries and radial profiles;
- Bayesian evidence and model comparison;
- residual maps, corner plots, and reports.

The guiding architecture is:

```text
simple notebook
      ↓
workflow / runner
      ↓
pipeline orchestration
      ↓
scientific core modules
```

Complex logic should not be implemented in notebooks.

## Current capabilities

- Single- and double-Sérsic models.
- Model-agnostic `render()` and `render_components()` interfaces.
- PSF-convolved modelling.
- Gaussian image likelihoods.
- Dynesty nested sampling with configurable convergence and parallelism.
- Bayesian evidence and Bayes-factor model comparison.
- Simulated, single-object real-image, and batch workflows.
- YAML-based reproducible configuration.
- Automatic target-specific image preparation.
- Fixed and automatic cutout modes.
- Elliptical radial-profile construction.
- Plateau-based automatic cutout sizing.
- Target-specific sky-region construction.
- Robust constant-background and MAD-based noise estimation.
- Empirical PSF candidate selection and stacking.
- Posterior radial-profile bands.
- Corner plots, residual maps, radial-profile diagnostics, and reports.
- Command-line project initialization and example export.

## Automatic image-preparation workflow

For `data.cutout.mode: auto`, BayMorph resolves the image region through the following sequence:

```text
full image
    ↓
preliminary source detection
    ↓
target selection near data.target_position
    ↓
target segment in full-image coordinates
    ↓
preliminary adaptive cutout
    ↓
second detection and target selection
    ↓
initial target geometry from segment moments
    ↓
elliptical radial profile
    ↓
profile plateau detection
    ↓
final cutout = plateau radius + sky margin
    ↓
final detection and masking
    ↓
final geometry, radial profile, and plateau
    ↓
sky region outside the target plateau
    ↓
constant background + robust noise estimate
```

The configured `target_position` is a **seed used to identify the target**. After final segmentation, BayMorph uses the measured target geometry as the final galaxy centre in cutout coordinates.

The legacy fixed-size mode remains available:

```yaml
data:
  cutout:
    mode: fixed
    size: 200
```

The recommended automatic mode is:

```yaml
data:
  cutout:
    mode: auto
    initial_factor: 4.0
    sky_margin: 20.0
    min_size: 100
    max_size: null
```

## Installation

BayMorph is designed to be installed inside a dedicated Conda environment.

Conda creates and manages an isolated Python environment, while `pip`
installs BayMorph and its Python dependencies inside that environment.

### 1. Create a Conda environment

```bash
conda create -n baymorph -c conda-forge python=3.11 -y
conda activate baymorph
```

### 2. Upgrade pip

```bash
python -m pip install --upgrade pip
```

### 3. Install BayMorph

Install the latest development version directly from GitHub:

```bash
python -m pip install \
"git+https://github.com/carlocannarozzophd/BayMorph.git@main"
```

### 4. Verify the installation

```bash
baymorph --version
baymorph --help
```

BayMorph supports **Python 3.10 and later**.

## Command-line interface

Check the installed version:

```bash
baymorph --version
```

Create a complete external project directory:

```bash
baymorph init MyBayMorphProject
```

This creates:

```text
MyBayMorphProject/
├── configs/
├── notebooks/
├── data/
└── runs/
```

Copy the packaged notebooks and YAML configurations without creating the full project tree:

```bash
baymorph examples BayMorphExamples
```

Use `--force` only when existing files should be overwritten:

```bash
baymorph examples BayMorphExamples --force
```

The copied files live outside the installed package, so users can edit them freely and update BayMorph without losing their work.

## Updating BayMorph

To update BayMorph to the latest version available on GitHub:

```bash
conda activate baymorph

python -m pip install \
    --upgrade \
    --force-reinstall \
    "git+https://github.com/carlocannarozzophd/BayMorph.git@main"
```

If you already have an existing BayMorph project, update the packaged
configuration files and notebooks with:

```bash
cd MyBayMorphProject

baymorph examples . --force
```

This updates only:

```text
configs/
notebooks/
```

while preserving your own:

```text
data/
runs/
```

and any scientific results already produced.

## Python quick start

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

For workflows requiring an empirical PSF:

```python
runner.prepare_image()
runner.select_psf_candidates()
runner.build_psf()
runner.fit()
runner.diagnostics()
```

Pipeline products are available through:

```python
runner.image
runner.psf_candidates
runner.psf
runner.results
```

To return a step result directly:

```python
products = runner.prepare_image(return_result=True)
fit_result = runner.fit(return_result=True)
```

## Official notebooks

| Notebook | Purpose |
| --- | --- |
| [`01_Simulated_Galaxy_Analysis.ipynb`](notebooks/01_Simulated_Galaxy_Analysis.ipynb) | Controlled end-to-end validation on a simulated galaxy. |
| [`02_Real_Galaxy_Analysis.ipynb`](notebooks/02_Real_Galaxy_Analysis.ipynb) | Complete workflow on a real galaxy. |
| [`03_Multi_Galaxy_Analysis.ipynb`](notebooks/03_Multi_Galaxy_Analysis.ipynb) | Batch processing of multiple FITS images. |

Installed users can copy these notebooks with `baymorph examples` or `baymorph init`.

## Model architecture

Every morphology model exposes:

```python
model.render(shape)
model.render_components(shape)
```

`render(shape)` returns the complete model image. `render_components(shape)` returns named components, for example:

```python
{
    "bulge": bulge_image,
    "disk": disk_image,
}
```

This common interface allows the fitting, posterior, diagnostics, and plotting code to remain model agnostic.

## Important flux convention

The current discrete Sérsic renderer normalizes `flux` over the finite rendered array. Therefore, in v0.2.0, `flux` is the flux represented within the requested rendering shape rather than an analytic infinite-radius Sérsic integral. This convention is internally consistent between simulation and fitting but is shape dependent. A future renderer may introduce pixel-integrated analytic normalization as a separate, validated change.

## Documentation

- [User and developer guide](docs/README.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Validation](docs/VALIDATION.md)
- [Architectural decisions](DECISIONS.md)
- [Roadmap](ROADMAP.md)

## Development checks

```bash
ruff check .
pytest
```

## License

BayMorph is distributed under the MIT License.
