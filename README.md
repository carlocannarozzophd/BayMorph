# BayMorph

<p align="center">
  <strong>Bayesian inference for astronomical morphology.</strong>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/python-%E2%89%A53.11-green">
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-MIT-lightgrey"></a>
  <a href="https://github.com/carlocannarozzophd/BayMorph/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/carlocannarozzophd/BayMorph/actions/workflows/ci.yml/badge.svg"></a>
  <img alt="Status" src="https://img.shields.io/badge/status-active%20development-orange">
</p>

<p align="center">
  <a href="#installation">Installation</a> ·
  <a href="#quick-start">Quick Start</a> ·
  <a href="notebooks/01_Simulated_SingleSersic_SingleBand_Fit.ipynb">Tutorial</a> ·
  <a href="docs/API.md">API</a> ·
  <a href="docs/architecture/ARCHITECTURE.md">Architecture</a> ·
  <a href="ROADMAP.md">Roadmap</a>
</p>

---

> **Need galaxy morphology?**
>
> **BayMorph.**
>
> Need Bayesian inference?
>
> **BayMorph.**
>
> Need a relaxing afternoon by the bay?
>
> Sorry. We only borrowed the name.

---

> [!IMPORTANT]
>
> BayMorph is under active scientific development.
>
> The current implementation provides physically consistent forward modelling,
> backend-neutral Bayesian inference, reproducible observation preparation,
> evidence-aware model comparison, posterior diagnostics, and scientific
> validation infrastructure.

## What is BayMorph?

BayMorph is an open-source scientific Python framework for Bayesian
astronomical morphology, statistical inference, simulation, and astronomical
image analysis.

Rather than treating galaxy fitting as the optimization of an analytical
profile evaluated at detector pixels, BayMorph models the observation process
connecting an intrinsic light distribution to measured detector data.

Its scientific architecture separates:

- intrinsic morphology;
- observation modelling;
- numerical rendering;
- probabilistic inference;
- inference backend;
- workflow orchestration;
- scientific validation.

Posterior distributions, parameter degeneracies, and Bayesian evidence are
first-class scientific products.

> **Galaxy morphology is fundamentally an inference problem, not merely an
> optimization problem.**

## Why BayMorph?

Morphology analyses commonly require separate tools for:

- image ingestion and preparation;
- source detection and segmentation;
- masking;
- PSF construction;
- forward modelling;
- Bayesian inference;
- posterior analysis;
- scientific diagnostics.

This fragmentation complicates reproducibility and makes scientific assumptions
harder to trace.

BayMorph is designed as one coherent framework in which scientific components
remain modular, explicit, independently testable, and reproducibly assembled
into complete analyses.

Configuration files define analyses.

Notebooks provide user-facing scientific workflows rather than hidden
implementations.

## Scientific scope

BayMorph currently focuses on:

- galaxy morphology;
- astronomical imaging;
- Bayesian statistical inference;
- physically motivated forward modelling;
- single-observation inference;
- correlated multi-observation inference;
- reproducible scientific workflows;
- scientific validation.

The current scope intentionally excludes:

- stellar-population modelling;
- spectral-energy-distribution fitting;
- dust modelling;
- radiative transfer.

## Current capabilities

| Area | Current capability |
| --- | --- |
| Observation preparation | FITS ingestion, target identification, automatic and fixed cutouts, background and noise estimation |
| Detection and masking | Source detection, segmentation, contaminant identification, fitting-mask construction |
| Morphology models | Single- and double-Sérsic models, composite models, analytic total-flux normalization |
| Rendering | Detector-pixel integration, extended-domain rendering, PSF convolution, adaptive integration infrastructure |
| PSF workflows | Analytic Gaussian PSFs and empirical PSF construction |
| Likelihoods | Gaussian likelihoods, with additional likelihood infrastructure under development |
| Inference | Backend-neutral inference problems with pocoMC and Dynesty production backends |
| Dynesty modes | Static and dynamic nested sampling |
| Posterior products | Weighted posterior samples, summaries, diagnostics, and canonical backend-neutral inference results |
| Evidence | Bayesian evidence and evidence-aware model comparison |
| Diagnostics | Model images, residuals, normalized residuals, radial profiles, posterior diagnostics, and reports |
| Validation | Mathematical, numerical, statistical, backend, and workflow validation infrastructure |
| Multi-observation architecture | Observation-centred representations and joint inference abstractions |

## Architecture

BayMorph is observation-centred and backend-neutral.

    Physical object
          |
          v
    Observation-specific intrinsic representation
          |
          v
    Observation operator
          |
          v
    Expected detector measurement
          |
          v
    Likelihood
          |
          v
    Inference problem
          |
          v
    Numerical backend
          |
          v
    InferenceResult

The probabilistic problem is defined independently of the numerical inference
backend.

Changing from pocoMC to Dynesty does not redefine the scientific model, prior,
likelihood, parameter meaning, or result interpretation.

The authoritative architectural documentation is available under
[`docs/architecture/`](docs/architecture/).

## Installation

BayMorph supports **Python 3.11 and later**.

Install directly from GitHub:

~~~bash
conda create -n baymorph -c conda-forge python=3.11 -y
conda activate baymorph

python -m pip install --upgrade pip
python -m pip install \
  "git+https://github.com/carlocannarozzophd/BayMorph.git@main"
~~~

Verify the installation:

~~~bash
baymorph --version
baymorph --help
~~~

The standard BayMorph installation includes the currently supported production
inference backends:

- pocoMC;
- Dynesty.

### Development installation

~~~bash
git clone https://github.com/carlocannarozzophd/BayMorph.git
cd BayMorph

python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
~~~

Run the quality checks with:

~~~bash
git diff --check
python -m ruff check .
python -m pytest -q
~~~

## Quick start

BayMorph analyses are driven by YAML configuration files.

The recommended high-level workflow is:

~~~python
from baymorph import BayMorphRunner

runner = BayMorphRunner("configs/simulated_galaxy.yaml")

result = runner.run(
    diagnostics=True,
)
~~~

`run()` performs the configured workflow, including image preparation,
conditional empirical-PSF construction, Bayesian fitting, and optional
diagnostics.

For interactive scientific work, the same workflow may be executed
step-by-step:

~~~python
from baymorph import BayMorphRunner

runner = BayMorphRunner("configs/simulated_galaxy.yaml")

runner.prepare_image()
runner.build_psf()
result = runner.fit()
runner.diagnostics()
~~~

For configurations using an empirical PSF, candidate selection precedes PSF
construction. The high-level `run()` workflow handles this automatically.

Resolved configurations and scientific products are written to a structured run
directory to preserve reproducibility.

## Inference backends

pocoMC is the default production inference backend.

A sampler configuration has a deliberately thin BayMorph envelope:

~~~yaml
sampler:
  name: pocomc
  init:
    n_effective: 512
    n_active: 256
    random_state: 42
  run:
    n_total: 4096
    n_evidence: 4096
    progress: true
~~~

Entries under `init` and `run` use the native names and semantics of the
selected upstream sampler.

BayMorph does not maintain a duplicate list of pocoMC or Dynesty numerical
options.

For the complete available options, consult the documentation corresponding to
the installed upstream package version:

- [pocoMC documentation](https://pocomc.readthedocs.io/)
- [Dynesty documentation](https://dynesty.readthedocs.io/)

### Dynesty static nested sampling

~~~yaml
sampler:
  name: dynesty
  mode: static
  init:
    nlive: 500
    sample: rwalk
  run:
    dlogz: 0.05
    print_progress: true
~~~

### Dynesty dynamic nested sampling

~~~yaml
sampler:
  name: dynesty
  mode: dynamic
  init:
    nlive: 500
    sample: rwalk
  run:
    nlive_init: 500
    nlive_batch: 250
    dlogz_init: 0.05
    print_progress: true
~~~

The numerical configuration belongs to the selected sampler.

The scientific inference problem does not.

## Inference results

Production inference returns the canonical backend-neutral `InferenceResult`.

The result separates:

- `PosteriorSamples`;
- `EvidenceEstimate`, when available;
- backend diagnostics;
- provenance.

Sampler-specific implementation state is intentionally not part of the stable
scientific result contract.

The sampler configuration supplied to the backend is recorded in inference
provenance and fit summaries.

## Learn BayMorph

The recommended starting point is:

    notebooks/01_Simulated_SingleSersic_SingleBand_Fit.ipynb

The tutorial demonstrates the complete scientific workflow from configuration
through posterior inference and diagnostics.

The observed single-observation workflow is available in:

    notebooks/02_Observed_Galaxy_Analysis.ipynb

Joint inference from correlated observations of the same astronomical object
is demonstrated in:

    notebooks/03_MultiObservation_ConfigMaker_and_Joint_Fit.ipynb

The public Config Maker helpers allow these workflows to select the morphology
model and inference sampler explicitly while keeping backend-specific machinery
out of the notebook.

## Documentation

| Document | Purpose |
| --- | --- |
| [Scientific documentation](docs/index.md) | Documentation landing page |
| [Architecture](docs/architecture/ARCHITECTURE.md) | Scientific and software architecture |
| [Inference](docs/architecture/INFERENCE.md) | Probabilistic formulation and backend contract |
| [Pipeline](docs/architecture/PIPELINE.md) | Workflow, configuration, provenance, and persistence |
| [Public API](docs/API.md) | Stable public interfaces |
| [Validation](docs/architecture/VALIDATION.md) | Validation architecture |
| [Roadmap](ROADMAP.md) | Current scientific and development priorities |
| [Changelog](CHANGELOG.md) | Release history |

## Development status

BayMorph is an active scientific research programme and an evolving Python
package.

Current priorities are:

1. probabilistic formulation;
2. BayMorph manuscript;
3. architecture;
4. implementation;
5. validation.

The native BayMorph nested sampler is archived and is not an active development
priority.

This archived sampler is distinct from the supported Dynesty backend.

The long-term scientific objective is coherent Bayesian inference from
correlated observations of the same underlying astronomical object.

## Contributing

Contributions are welcome.

Changes affecting scientific models, priors, likelihoods, rendering, inference,
results, or public interfaces must include corresponding validation and
synchronized documentation.

Before committing code changes, run:

~~~bash
git diff --check
python -m ruff check .
python -m pytest -q
~~~

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the complete development workflow.

## Citation

Citation metadata are provided in [`CITATION.cff`](CITATION.cff).

When using a development version of BayMorph in scientific work, record:

- the BayMorph version;
- the Git commit;
- the resolved configuration;
- input-data provenance;
- inference configuration;
- rendering configuration.

Citation information for the BayMorph paper will be added when the manuscript
is released.

## License

BayMorph is distributed under the MIT License.

See [`LICENSE`](LICENSE) for details.
