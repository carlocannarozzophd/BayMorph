# BayMorph Documentation

Welcome to the BayMorph documentation.

BayMorph is a modern Python library for Bayesian galaxy morphology modelling from astronomical imaging. The documentation is organized into a small number of guides covering the architecture, scientific workflows, validation strategy, and future development of the project.

---

# Getting Started

If you are new to BayMorph, begin with:

1. **Project overview**
   - [`../README.md`](../README.md)

2. **Installation**
   - Install BayMorph inside a dedicated Conda environment.
   - Create an external project with `baymorph init`.
   - Copy packaged examples with `baymorph examples`.

3. **Run your first analysis**
   - Open one of the official notebooks.
   - Load a configuration file.
   - Create a `BayMorphRunner`.
   - Execute the workflow.

---

# Documentation Guide

## Architecture

**[`ARCHITECTURE.md`](ARCHITECTURE.md)**

Describes the overall software architecture, package responsibilities, workflow organization, configuration philosophy, and design principles.

Read this document if you want to understand how BayMorph is internally organized.

---

## Validation

**[`VALIDATION.md`](VALIDATION.md)**

Describes the scientific and software validation strategy, including

- automated testing;
- regression tests;
- simulated galaxies;
- real observations;
- code-quality checks.

---

## API Guide

**[`API.md`](API.md)**

Describes the public, advanced, and internal APIs.

---

## Architectural Decisions

**[`../DECISIONS.md`](../DECISIONS.md)**

Chronological record of important architectural decisions together with their rationale.

---

## Multi-Observation Inference

**[`architecture/MULTI_OBSERVATION_PUBLIC_WORKFLOW.md`](architecture/MULTI_OBSERVATION_PUBLIC_WORKFLOW.md)**

Defines the public workflow for Bayesian inference from correlated observations
of the same astronomical object.

The underlying probabilistic formulation is developed in
[`research/MULTIBAND_FORMULATION.md`](research/MULTIBAND_FORMULATION.md).
The historical filename reflects legacy implementation terminology; the
scientific abstraction is multi-observation rather than intrinsically
multi-band.

---

## Roadmap

**[`../ROADMAP.md`](../ROADMAP.md)**

Current development priorities together with long-term project goals.

---

# Typical Workflow

The standard workflow is

```text
Configuration
        │
        ▼
Image Preparation
        │
        ▼
PSF Construction
        │
        ▼
Bayesian Fitting
        │
        ▼
Diagnostics
```

The high-level notebook interface is provided by

```python
BayMorphRunner
```

which orchestrates the complete scientific workflow.

---

# Official Examples

BayMorph includes three canonical public notebooks covering

- a simulated single-observation fit;
- an observed single-observation analysis;
- a correlated multi-observation joint fit.

Install them with

```bash
baymorph examples DESTINATION
```

or

```bash
baymorph init PROJECT_DIRECTORY
```

---

# Development

See [`../CONTRIBUTING.md`](../CONTRIBUTING.md) for the complete contribution guide.

Contributors should run

```bash
ruff check .

python -m pytest
```

before submitting changes.

---

# Project Philosophy

BayMorph is designed around a simple principle:

> **Every architectural layer owns exactly one responsibility.**

Scientific algorithms remain reusable, workflow orchestration remains reproducible, and notebooks remain lightweight user interfaces rather than implementation code.