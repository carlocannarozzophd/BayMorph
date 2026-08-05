# BayMorph Public API

## Purpose

This document defines the public software interfaces of BayMorph.

Its purpose is not to document every Python object.

Instead, it defines:

- which interfaces are intended for public use;
- which interfaces are considered internal;
- the stability guarantees associated with each interface;
- the principles governing API evolution.

Detailed API reference documentation is generated separately.

---

# API Philosophy

The BayMorph API follows the same principles as the scientific architecture.

## Scientific interfaces first

Public interfaces represent scientific concepts rather than implementation
details.

Users should interact with concepts such as observations, morphology models,
inference problems, and posterior results rather than internal implementation
classes.

---

## Backend neutrality

Public interfaces must remain independent of the inference backend.

Changing the backend must not require users to rewrite scientific workflows.

Backend-specific objects are implementation details.

---

## Stable scientific semantics

Scientific meaning has higher priority than implementation stability.

Implementations may evolve.

Scientific interpretation should not.

A parameter should retain the same scientific meaning even if its internal
implementation changes.

---

## Minimal public surface

Only interfaces with long-term scientific meaning should become part of the
public API.

Internal implementation details should remain private.

A small stable API is preferable to a large unstable one.

---

# Design Principles

The public API follows these principles.

- Explicit over implicit.
- Composition over duplication.
- Scientific correctness before convenience.
- Stable semantics before stable implementation.
- Backend-neutral scientific workflows.
- Clear separation between public and internal interfaces.

---

# Public API

The following categories constitute the supported public API.

## User workflows

High-level user interaction is performed through

- `BayMorphRunner`

This represents the recommended entry point for complete scientific analyses.

---

## Scientific data

Public scientific data structures include

- `Observation`
- `ObservationMetadata`
- `ImageData`

These objects describe astronomical observations rather than implementation
details.

---

## Morphology models

Current public morphology models include

- `SersicModel`
- `DoubleSersicModel`
- `CompositeModel`

Future morphology families should integrate into the same public model
architecture.

---

## Point-spread functions

Current public PSF interfaces include

- `GaussianPSF`
- `EmpiricalPSF`

Additional PSF implementations should preserve the same conceptual interface.

---

## Scientific results

Current public result objects include

- `PosteriorResult`

Future result containers should extend rather than replace this abstraction
whenever possible.

---

# Stable Scientific Interfaces

The public API is organized around stable scientific interfaces.

Conceptually, BayMorph follows the sequence

    Observation
          |
          v
    Representation
          |
          v
    Observation Operator
          |
          v
    Likelihood
          |
          v
    Inference Problem
          |
          v
    Posterior Result

These interfaces define scientific responsibilities.

Concrete implementations may evolve while preserving this conceptual contract.

---

# Extension Points

BayMorph is designed to be extended through composition.

Typical extension points include

- morphology models;
- observation operators;
- likelihoods;
- rendering algorithms;
- inference backends;
- posterior diagnostics;
- validation experiments.

Extensions should reuse existing abstractions whenever possible.

New abstractions should be introduced only when multiple scientific use cases
justify them.

---

# Internal APIs

Many modules are intentionally **not** part of the public API.

Examples include internal implementation details within

- `baymorph.fitting`
- `baymorph.rendering`
- `baymorph.pipeline`
- `baymorph.validation`

These modules may evolve without preserving backward compatibility.

Users should avoid depending directly on internal implementation details unless
they are actively developing BayMorph itself.

---

# API Stability Policy

BayMorph distinguishes four stability levels.

| Status | Meaning |
| ------- | ------- |
| **Stable** | Backward compatibility is expected across minor releases. |
| **Experimental** | Interface may evolve as the scientific architecture matures. |
| **Internal** | No compatibility guarantees are provided. |
| **Deprecated** | Interface is scheduled for removal after an appropriate transition period. |

Unless explicitly documented otherwise, newly introduced scientific interfaces
should initially be considered **Experimental**.

---

# Compatibility Policy

Backward compatibility is maintained for stable public interfaces whenever
scientifically reasonable.

Compatibility must not preserve scientifically incorrect behaviour.

If scientific correctness requires changing an interface, the transition should
be explicit and documented.

Internal APIs may change at any time.

---

# Deprecation Policy

Public interfaces are not removed abruptly.

The preferred sequence is

1. introduce the replacement;
2. mark the previous interface as deprecated;
3. document the migration path;
4. remove the deprecated interface in a future release.

Deprecated interfaces should remain functional during the transition period
whenever practical.

---

# Versioning

Software releases describe implementation evolution.

Architectural evolution is described by

- the constitutional architecture;
- specialized architecture documents;
- Architecture Decision Records.

These documents define the long-term scientific contract independently of
release numbering.

---

# Relationship to the Architecture

The public API is an implementation of the BayMorph scientific architecture.

Its design follows

- `docs/architecture/CORE_ARCHITECTURE.md`
- `docs/architecture/OBSERVATION_MODEL.md`
- `docs/architecture/INFERENCE.md`
- `docs/architecture/RENDERING.md`
- `docs/architecture/PIPELINE.md`
- `docs/architecture/VALIDATION.md`

Architecture defines scientific responsibilities.

The API exposes those responsibilities to users.

---

# Future Evolution

Future public interfaces will continue to be guided by

- the probabilistic formulation;
- the observation-centred architecture;
- backend neutrality;
- validated numerical methods;
- reproducible scientific workflows.

The public API should evolve conservatively.

Scientific meaning should remain stable even as implementation details improve.
