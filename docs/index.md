# BayMorph Scientific Documentation

<p align="center">
  <strong>Bayesian inference for astronomical morphology.</strong>
</p>

BayMorph is an open-source scientific Python framework for Bayesian
astronomical morphology, statistical inference, simulation, and astronomical
image analysis.

Its central scientific principle is:

> **Galaxy morphology is fundamentally an inference problem, not merely an
> optimization problem.**

BayMorph connects physically interpretable morphology models, explicit
observation operators, backend-neutral Bayesian inference, validated numerical
rendering, and reproducible scientific workflows.

## Scientific philosophy

> **Theory drives architecture.**
>
> **Architecture drives implementation.**
>
> **Implementation validates theory.**
>
> **Scientific correctness has priority over implementation convenience.**

BayMorph is developed simultaneously as:

- a scientific research programme;
- a Bayesian inference framework;
- a scientific Python package;
- a validation framework;
- a scientific manuscript.

## Scientific architecture

BayMorph is observation-centred.

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

Single-observation inference is the one-observation specialization of the same
architecture used for multiple observations of the same physical object.

Inference backends evaluate the probabilistic problem but do not define it.

[Explore the architecture](architecture/ARCHITECTURE.md)

## Production inference

BayMorph currently provides two production inference backends:

- **pocoMC**, the default backend;
- **Dynesty**, with both static and dynamic nested sampling.

Sampler-specific numerical arguments retain the names and semantics defined by
the upstream libraries.

BayMorph does not maintain a duplicate sampler API.

[Read the inference architecture](architecture/INFERENCE.md)

## Canonical scientific results

Production inference returns the backend-neutral `InferenceResult`.

The result contract separates:

- weighted posterior samples;
- Bayesian evidence, when available;
- diagnostics;
- inference provenance.

Backend-specific internal sampler state is not part of the scientific result
contract.

[Read the public API](API.md)

## Documentation map

<div class="grid cards" markdown>

-   **Core Architecture**

    ---

    Constitutional principles, scientific objects, invariants, conceptual
    layers, and evolution rules.

    [Read the constitution](architecture/CORE_ARCHITECTURE.md)

-   **Observation Model**

    ---

    Observations, intrinsic representations, observation operators, parameter
    routing, and cross-observation coupling.

    [Read the observation model](architecture/OBSERVATION_MODEL.md)

-   **Inference**

    ---

    Inferential states, priors, likelihoods, backend configuration, posterior
    products, evidence, and backend neutrality.

    [Read the inference architecture](architecture/INFERENCE.md)

-   **Rendering**

    ---

    Detector integration, coordinate semantics, finite support, PSF
    convolution, and numerical error control.

    [Read the rendering architecture](architecture/RENDERING.md)

-   **Pipeline**

    ---

    Configuration, observation preparation, problem construction, execution,
    persistence, provenance, and reporting.

    [Read the pipeline architecture](architecture/PIPELINE.md)

-   **Validation**

    ---

    Mathematical, numerical, statistical, backend, and workflow validation.

    [Read the validation architecture](architecture/VALIDATION.md)

-   **Public API**

    ---

    Stable public interfaces, canonical result objects, compatibility, and
    extension policy.

    [Read the API policy](API.md)

-   **Research Notes**

    ---

    Mathematical formulations, numerical research, and manuscript planning.

    [Explore the research documentation](research/PAPER_BLUEPRINT.md)

</div>

## Scientific scope

BayMorph currently focuses on:

- galaxy morphology;
- astronomical imaging;
- Bayesian statistical inference;
- physically motivated forward modelling;
- correlated observations of the same astronomical object;
- reproducible scientific workflows;
- scientific validation.

The current scope excludes:

- stellar-population modelling;
- spectral-energy-distribution fitting;
- dust modelling;
- radiative transfer.

## Current priorities

Current development follows this order:

1. probabilistic formulation;
2. BayMorph manuscript;
3. architecture;
4. implementation;
5. validation.

The native BayMorph nested sampler is archived and is not an active development
priority.

This archived sampler is distinct from the supported Dynesty production
backend.

## Project links

- [Repository](https://github.com/carlocannarozzophd/BayMorph)
- [Architecture](architecture/ARCHITECTURE.md)
- [Inference](architecture/INFERENCE.md)
- [Public API](API.md)
- [Roadmap](https://github.com/carlocannarozzophd/BayMorph/blob/main/ROADMAP.md)
- [Contributing](https://github.com/carlocannarozzophd/BayMorph/blob/main/CONTRIBUTING.md)
