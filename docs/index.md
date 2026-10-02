# BayMorph Documentation

<p align="center">
  <strong>Bayesian inference for astronomical morphology.</strong>
</p>

BayMorph is an open-source Python framework for Bayesian inference of galaxy
morphology from one or more observations of the same object. It connects
interpretable morphology models, explicit observation operators,
backend-neutral Bayesian inference, validated numerical rendering and
reproducible workflows.

> **Galaxy morphology is fundamentally an inference problem, not merely an
> optimization problem.**

!!! warning "Status"

    BayMorph is under active scientific development. The observation model
    of the real-data path is being revised where the first real fit measured
    defects: the effective-PSF rendering of AD-046 is implemented, and the
    new noise model of AD-047 is accepted and being implemented. No
    real-data fit has yet been validated scientifically. The order of the
    remaining work is stated in the
    [roadmap](https://github.com/carlocannarozzophd/BayMorph/blob/main/ROADMAP.md).

## Start here

<div class="grid cards" markdown>

-   **Getting Started**

    ---

    Installation, a first analysis, the command line, and the tutorial
    notebooks.

    [Quick start](guide/QUICK_START.md)

-   **Configuration Reference**

    ---

    Every section and key of an analysis configuration, with its default and
    what a change invalidates.

    [Configuration](guide/CONFIGURATION.md)

-   **Models**

    ---

    The morphology models and their components, the PSFs and the
    residual background, with their parameters, priors and validation.

    [Models](models/index.md)

-   **Public API**

    ---

    The stable Python interfaces, the canonical result objects and the
    stability policy.

    [Public API](API.md)

-   **Architecture**

    ---

    The scientific contracts: observation model, rendering, inference,
    pipeline and validation.

    [Architecture overview](architecture/ARCHITECTURE.md)

-   **Research Notes**

    ---

    The measurements and formulations behind the architecture, grouped by
    workstream with their status.

    [Research overview](research/index.md)

-   **Decisions**

    ---

    The accepted architectural decision records, AD-001 onwards.

    [DECISIONS.md](https://github.com/carlocannarozzophd/BayMorph/blob/main/DECISIONS.md)

</div>

## Scientific architecture

BayMorph is observation-centred:

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

Single-observation inference is the one-observation case of the architecture
used for several observations of the same object. Inference backends evaluate
the probabilistic problem but do not define it.

## Inference backends

Nautilus is the default backend; Dynesty, with static or dynamic nested
sampling, and pocoMC are supported alternatives. Their numerical options keep
the names and meaning of the upstream libraries. Every backend returns the
backend-neutral `InferenceResult`, whose weighted posterior samples are the
statistical source of truth.

## Scope

BayMorph covers galaxy morphology from astronomical images, single- and
multi-observation Bayesian inference with conditional independence across
observations stated explicitly, and the validation of both. It does not
cover stellar populations, spectral-energy-distribution fitting, dust,
radiative transfer, residual covariance between observations, or
survey-level population inference.

## Project links

- [Repository](https://github.com/carlocannarozzophd/BayMorph)
- [Roadmap](https://github.com/carlocannarozzophd/BayMorph/blob/main/ROADMAP.md)
- [Changelog](https://github.com/carlocannarozzophd/BayMorph/blob/main/CHANGELOG.md)
- [Contributing](https://github.com/carlocannarozzophd/BayMorph/blob/main/CONTRIBUTING.md)
