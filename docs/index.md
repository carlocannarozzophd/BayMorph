# BayMorph Scientific Documentation

<p align="center">
  <strong>Bayesian inference for astronomical morphology.</strong>
</p>

BayMorph is an open-source scientific Python framework for Bayesian
astronomical morphology, statistical inference, and scientific image analysis.

It is designed around a simple principle:

> **Galaxy morphology is fundamentally an inference problem, not merely an
> optimization problem.**

BayMorph connects physically interpretable morphology models, explicit
observation operators, backend-neutral Bayesian inference, validated numerical
rendering, and reproducible scientific workflows.

---

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

These components evolve together.

---

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
    Posterior inference
          |
          v
    Scientific result

Single-observation inference is the one-observation specialization of the same
architecture used for multiple observations of the same physical object.

Inference backends evaluate the probabilistic problem but do not define it.

[Explore the architecture](architecture/ARCHITECTURE.md)

---

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

    Inferential states, priors, likelihoods, posterior products, evidence, and
    backend neutrality.

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

    Public interfaces, extension points, stability levels, compatibility, and
    deprecation policy.

    [Read the API policy](API.md)

-   **Research Notes**

    ---

    Mathematical formulations, numerical research, and manuscript planning.

    [Explore the research documentation](research/PAPER_BLUEPRINT.md)

</div>

---

## Current development

Current development focuses on:

1. completing the production multi-observation workflow;
2. migrating `real_fit.py` to the observation-centred inference architecture;
3. consolidating scientific documentation;
4. validating joint inference on representative astronomical observations.

The native nested sampler is archived and is not an active development
priority.

---

## Scientific scope

BayMorph currently focuses on:

- galaxy morphology;
- astronomical imaging;
- Bayesian statistical inference;
- physically motivated forward modelling;
- correlated multi-observation inference;
- reproducible scientific workflows.

The current scope excludes stellar-population modelling, SED fitting, dust
modelling, and radiative transfer.

---

## Project links

- [Repository](https://github.com/carlocannarozzophd/BayMorph)
- [Architecture](architecture/ARCHITECTURE.md)
- [Public API](API.md)
- [Contributing](https://github.com/carlocannarozzophd/BayMorph/blob/feature/baymorph-v0.4/CONTRIBUTING.md)
- [Roadmap](https://github.com/carlocannarozzophd/BayMorph/blob/feature/baymorph-v0.4/ROADMAP.md)
