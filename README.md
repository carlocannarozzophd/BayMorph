# BayMorph

<p align="center">
  <strong>Bayesian inference for astronomical morphology.</strong>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/python-%E2%89%A53.11-green">
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-MIT-lightgrey"></a>
  <img alt="Status" src="https://img.shields.io/badge/status-active%20development-orange">
</p>

<p align="center">
  <a href="https://carlocannarozzophd.github.io/BayMorph/">Documentation</a> ·
  <a href="#installation">Installation</a> ·
  <a href="#quick-start">Quick Start</a> ·
  <a href="docs/API.md">API</a> ·
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
> BayMorph is under active scientific development. The observation model of
> the real-data path is being revised where the first real fit measured
> defects: the effective-PSF rendering of AD-046 is implemented, and the new
> noise model of AD-047 is accepted but not yet implemented. No real-data fit
> has yet been validated scientifically. The order of the remaining work is
> in the [roadmap](ROADMAP.md).

## What is BayMorph?

BayMorph is an open-source Python framework for Bayesian inference of galaxy
morphology from one or more observations of the same object.

Rather than treating galaxy fitting as the optimization of an analytical
profile evaluated at detector pixels, BayMorph models the observation process
that connects an intrinsic light distribution to the measured detector data.
Its architecture keeps separate the intrinsic morphology, the observation
model, the numerical rendering, the probabilistic inference, the inference
backend, the workflow and the validation. Posterior distributions, parameter
degeneracies and Bayesian evidence are its scientific products.

> **Galaxy morphology is fundamentally an inference problem, not merely an
> optimization problem.**

## Why BayMorph?

Morphology analyses commonly need separate tools for image preparation,
source detection, masking, PSF construction, forward modelling, Bayesian
inference and diagnostics, which makes their assumptions hard to trace.
BayMorph assembles these steps in one framework whose components stay
modular, explicit and independently tested. Configuration files define
analyses; notebooks are user workflows, not hidden implementations.

## Scientific scope

BayMorph covers galaxy morphology from astronomical images, single- and
multi-observation Bayesian inference with conditional independence across
observations stated explicitly, reproducible workflows and their
validation. It does not cover stellar populations, spectral-energy-
distribution fitting, dust, radiative transfer, residual covariance between
observations, or survey-level population inference.

## Current capabilities

| Area | Capability |
| --- | --- |
| Observation preparation | FITS ingestion, target identification, automatic and fixed cutouts, constant background subtraction, a robust sky noise scale (AD-040, AD-041), exclusion of impossibly negative pixels (AD-043) |
| Detection and masking | Source detection and segmentation, contaminant masking, external masks |
| Morphology models | Single Sérsic, two labelled Sérsic components, bulge and disc, Sérsic plus point source |
| Rendering | Detector-pixel integration, and effective-PSF rendering of empirical PSFs with sub-pixel kernels (AD-046) |
| PSF | Analytic Gaussian PSFs; empirical PSFs built from the stars of the image, with sub-pixel kernels and extended wings (AD-046) |
| Likelihoods | Normalized Gaussian likelihood with the prepared noise scale in both workflows; within-observation correlated Gaussian likelihood in the multi-observation workflow only. The noise model of the single path excludes the source Poisson variance and the correlation of the sky, which AD-047, accepted and not yet implemented, adds |
| Inference | Backend-neutral inference problems with Nautilus (default), Dynesty (static and dynamic nested sampling) and pocoMC |
| Results | Weighted posterior samples, Bayesian evidence, diagnostics and provenance in a backend-neutral `InferenceResult` |
| Multi-observation inference | One joint workflow with explicit coupling across observations (shared, independent, hierarchical, wavelength-dependent) and conditionally factorized likelihoods |
| Validation | Mathematical, numerical, statistical, backend and workflow validation infrastructure, including simulation-based calibration |
| Model screening | Experimental simulation-trained ranking, frozen; the choice of model remains the user's |

## Architecture

BayMorph is observation-centred and backend-neutral:

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

Changing among Nautilus, Dynesty and pocoMC does not change the scientific
model, the prior, the likelihood, the meaning of a parameter or the
interpretation of a result. The architectural contracts are in
[`docs/architecture/`](docs/architecture/ARCHITECTURE.md).

## Installation

BayMorph requires Python 3.11 or later:

~~~bash
conda create -n baymorph -c conda-forge python=3.11 -y
conda activate baymorph

python -m pip install --upgrade pip
python -m pip install "git+https://github.com/carlocannarozzophd/BayMorph.git@main"

baymorph --version
~~~

The installation includes the three supported inference backends. BayMorph
distributes software, example configurations, tutorial notebooks and tests.
It does not include observational data, and it does not include generated
results such as run directories, posteriors, benchmark tables or figures.
The development installation is described in
[`CONTRIBUTING.md`](CONTRIBUTING.md).

## Quick start

~~~bash
baymorph init my_analysis
cd my_analysis
baymorph run configs/simulated_galaxy.yaml
~~~

`init` creates a workspace with the example configuration and the tutorial
notebooks; `run` simulates a galaxy, prepares the image, fits it and writes
the diagnostics under `runs/`. For an observed image, generate a
configuration, review it, and run it:

~~~bash
baymorph config data/galaxy.fits \
  --target-x 500 --target-y 500 \
  --model single_sersic \
  --uniform-prior flux 100 1000000 \
  --output configs/galaxy.yaml

baymorph run configs/galaxy.yaml
~~~

The same workflow from Python:

~~~python
from baymorph import BayMorphRunner

runner = BayMorphRunner("configs/galaxy.yaml")
result = runner.run(diagnostics=True)
~~~

The [quick start](docs/guide/QUICK_START.md) explains each step, the
[command-line reference](docs/guide/CLI.md) lists every option, and the
[configuration reference](docs/guide/CONFIGURATION.md) every key, including
the sampler settings, which keep the native names of each backend.

## Documentation

The documentation site is <https://carlocannarozzophd.github.io/BayMorph/>.
Its sources are under [`docs/`](docs/index.md):

| Document | Purpose |
| --- | --- |
| [Getting Started](docs/guide/QUICK_START.md) | Installation, quick start, command line, configuration, products and tutorials |
| [Models](docs/models/index.md) | Morphology models and their components, PSFs and residual background, with their parameters and validation |
| [Public API](docs/API.md) | Stable public interfaces and result objects |
| [Architecture](docs/architecture/ARCHITECTURE.md) | Scientific contracts of observation, rendering, inference, pipeline and validation |
| [Research notes](docs/research/index.md) | Measurements and formulations behind the decisions |
| [Decisions](DECISIONS.md) | Accepted architectural decision records |
| [Roadmap](ROADMAP.md) | Current priority and remaining work |
| [Changelog](CHANGELOG.md) | Release history |

To build the site locally:

~~~bash
python -m pip install -r requirements-docs.txt
python -m mkdocs serve
~~~

## Development status

BayMorph is a scientific research programme and an evolving Python package.
The current priority and the order of the remaining work are stated in the
[roadmap](ROADMAP.md). The experimental native nested sampler is archived;
production sampling is delegated to established backends (AD-039). The
long-term objective is coherent Bayesian inference from several observations
of the same object, with every dependence assumption stated explicitly.

## Contributing

Contributions are welcome. Changes to scientific models, priors,
likelihoods, rendering, inference, results or public interfaces must come
with their validation and synchronized documentation. See
[`CONTRIBUTING.md`](CONTRIBUTING.md).

## Citation

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). When using a
development version of BayMorph in scientific work, record the BayMorph
version, the Git commit, the resolved configuration, the provenance of the
input data, and the inference and rendering configuration. Citation
information for the BayMorph paper will be added when the manuscript is
released.

## License

BayMorph is distributed under the MIT License; see [`LICENSE`](LICENSE).
