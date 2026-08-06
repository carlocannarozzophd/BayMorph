# BayMorph

<p align="center">
  <strong>Bayesian inference for astronomical morphology.</strong>
</p>

<p align="center">
  <a href="https://github.com/carlocannarozzophd/BayMorph/releases"><img alt="Version" src="https://img.shields.io/badge/version-v0.4-blue"></a>
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
> Sorry.
> We only borrowed the name.

---

> [!IMPORTANT]
>
> BayMorph is under active scientific development.
>
> The current implementation provides physically consistent forward modelling,
> backend-neutral Bayesian inference, evidence-based model comparison,
> reproducible observation preparation, posterior diagnostics, and a scientific
> validation framework.

## Contents

- [What is BayMorph?](#what-is-baymorph)
- [Why BayMorph?](#why-baymorph)
- [Scientific Philosophy](#scientific-philosophy)
- [Scientific Scope](#scientific-scope)
- [Current Capabilities](#current-capabilities)
- [Architecture](#architecture)
- [Documentation](#documentation)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Learn BayMorph](#learn-baymorph)
- [Development Status](#development-status)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

# What is BayMorph?

BayMorph is an open-source scientific Python framework for Bayesian
astronomical morphology, statistical inference, and scientific image analysis.

Rather than treating galaxy fitting as the optimization of an analytical
profile sampled at detector pixels, BayMorph models the observation process
connecting an intrinsic light distribution to measured detector data.

Its scientific architecture separates:

- intrinsic morphology;
- observation modelling;
- numerical rendering;
- statistical inference;
- workflow orchestration;
- scientific validation.

Posterior distributions, parameter degeneracies, and Bayesian evidence are
first-class scientific products.

> **Galaxy morphology is fundamentally an inference problem, not merely an optimization problem.**

# Why BayMorph?

Modern morphology analyses often require several independent tools for:

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

Notebooks remain user interfaces rather than hidden implementations.

# Scientific Philosophy

BayMorph is developed according to four principles:

> **Theory drives architecture.**
>
> **Architecture drives implementation.**
>
> **Implementation validates theory.**
>
> **Scientific correctness has priority over implementation convenience.**

These principles govern the probabilistic formulation, software architecture,
implementation, validation, documentation, and manuscript.

# Scientific Scope

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

These topics require additional scientific architecture and are not part of the
present BayMorph programme.

# Current Capabilities

| Area | Current capability |
| --- | --- |
| Observation preparation | FITS ingestion, target identification, automatic and fixed cutouts, background and noise estimation |
| Detection and masking | Source detection, segmentation, connected-source masking, optional refinement |
| Morphology models | Continuous single- and double-Sérsic models, composite models, analytic total-flux normalization |
| Rendering | Detector-pixel integration, extended-domain rendering, physically consistent PSF convolution, adaptive integration infrastructure |
| PSF workflows | Analytic Gaussian PSFs, empirical candidates, recentering, validation, stacking, normalization, persistence |
| Likelihoods | Gaussian likelihoods, with Cash and Poisson-Gaussian infrastructure |
| Inference | Backend-neutral inference problems with a production Dynesty backend |
| Posterior products | Posterior summaries, weighted and equal-weight samples, derived diagnostics |
| Evidence | Bayesian evidence, Bayes factors, and evidence-aware model comparison |
| Diagnostics | Corner plots, model and residual images, normalized residuals, radial profiles, preparation and runtime reports |
| Validation | Mathematical, numerical, statistical, backend, and workflow validation infrastructure |
| Multi-observation architecture | Observation-centred representations and joint inference abstractions |
| Production multi-observation workflow | Under active development |

## Forward-modelling principles

BayMorph treats a galaxy as a continuous distribution of light.

Detector pixels represent integrated signal over finite areas, not point samples.
The observation model may include coordinate transformations, pixel integration,
finite support, and PSF convolution before likelihood evaluation.

This preserves the scientific meaning of quantities such as total flux,
effective radius, centroid, axis ratio, and position angle independently of the
chosen inference backend.

# Architecture

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
    Posterior inference
          |
          v
    Scientific result

Single-observation inference is the one-observation specialization of the same
general architecture used for multiple observations of the same physical
object.

Cross-observation parameter sharing, observation-specific freedom, deterministic
relations, and hierarchical coupling are distinct probabilistic models and must
remain explicit.

The authoritative architectural documentation is available in
[`docs/architecture/`](docs/architecture/).

# Documentation

| Document | Purpose |
| --- | --- |
| [Architecture entry point](docs/architecture/ARCHITECTURE.md) | Navigation across the BayMorph scientific architecture |
| [Core Architecture](docs/architecture/CORE_ARCHITECTURE.md) | Constitutional principles and architectural invariants |
| [Observation Model](docs/architecture/OBSERVATION_MODEL.md) | Observation-centred scientific model |
| [Inference Architecture](docs/architecture/INFERENCE.md) | Priors, likelihoods, posterior inference, evidence, and backend neutrality |
| [Rendering Architecture](docs/architecture/RENDERING.md) | Detector integration, finite support, PSF operations, and numerical accuracy |
| [Pipeline Architecture](docs/architecture/PIPELINE.md) | Configuration, orchestration, persistence, provenance, and reporting |
| [Validation Architecture](docs/architecture/VALIDATION.md) | Mathematical, numerical, statistical, and workflow validation |
| [Architecture Decision Records](docs/architecture/decisions/README.md) | Governance and history of long-lived architectural decisions |
| [Public API](docs/API.md) | Stable public interfaces |
| [Roadmap](ROADMAP.md) | Scientific and development milestones |
| [Contributing](CONTRIBUTING.md) | Development workflow and contribution requirements |
| [Changelog](CHANGELOG.md) | Release history |

# Installation

BayMorph currently supports **Python 3.11 and later**.

Because the repository is private, installation requires GitHub access.

## Install from GitHub

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

## Development installation

```bash
git clone https://github.com/carlocannarozzophd/BayMorph.git
cd BayMorph

python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Run the quality checks:

```bash
git diff --check
python -m ruff check .
python -m pytest -q
```

# Quick Start

BayMorph analyses are driven by YAML configuration files.

```python
from baymorph import BayMorphRunner

runner = BayMorphRunner("configs/simulated_galaxy.yaml")

runner.prepare_image()
runner.fit()
runner.diagnostics()
```

The principal products remain accessible from the runner:

```python
prepared_image = runner.image
posterior = runner.results
profile_bands = runner.profile_bands
```

For empirical PSF workflows:

```python
runner.prepare_image()
runner.select_psf_candidates()
runner.build_psf()
runner.fit()
runner.diagnostics()
```

Resolved configurations and scientific products are written to a structured run
directory to preserve reproducibility.

# Learn BayMorph

The recommended starting point is:

```text
notebooks/01_Simulated_SingleSersic_SingleBand_Fit.ipynb
```

The tutorial demonstrates:

1. YAML configuration;
2. synthetic galaxy generation;
3. observation preparation;
4. background estimation;
5. noise estimation;
6. Gaussian PSF construction;
7. Bayesian fitting;
8. posterior interpretation;
9. scientific diagnostics.

# Development Status

BayMorph is an active scientific research programme and an evolving Python
package.

## Completed foundations

- observation-centred architecture;
- backend-neutral inference architecture;
- single-observation inference problems;
- joint inference abstractions;
- observation-indexed representation construction;
- adaptive numerical rendering infrastructure;
- Dynesty production backend;
- backend-neutral posterior products;
- configuration-driven workflows;
- scientific validation framework;
- constitutional architecture documentation.

## Current priorities

1. complete the production multi-observation workflow;
2. resume development of `real_fit.py`;
3. consolidate the public architecture and implementation documentation;
4. introduce Material for MkDocs as the official BayMorph scientific documentation;
5. validate correlated multi-observation inference on realistic astronomical data.

The native nested sampler is archived and is not an active development priority.

The long-term scientific objective is coherent Bayesian inference from
correlated observations of the same astronomical object.

See [`ROADMAP.md`](ROADMAP.md) for the authoritative development plan.

# Contributing

Contributions are welcome.

Changes affecting scientific models, priors, likelihoods, rendering, inference,
posterior products, or public interfaces must include corresponding validation
and synchronized documentation.

Before committing code changes, run:

```bash
git diff --check
python -m ruff check .
python -m pytest -q
```

Documentation-only changes normally require:

```bash
git diff --check
```

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the complete development workflow.

# Citation

Citation metadata are provided in [`CITATION.cff`](CITATION.cff).

When using a development version of BayMorph in scientific work, record:

- the BayMorph version;
- the Git commit;
- the resolved configuration;
- the input-data provenance;
- the inference and rendering configuration.

Citation information for the BayMorph paper will be added when the manuscript is
released.

# License

BayMorph is distributed under the MIT License.

See [`LICENSE`](LICENSE) for details.
