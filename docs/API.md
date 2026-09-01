# BayMorph Public API

## Purpose

This document defines the stable user-facing software interfaces of BayMorph.

It does not attempt to document every Python object. Its purpose is to define:

- which interfaces are intended for public use;
- which interfaces are implementation details;
- how scientific semantics are separated from inference backends;
- which result objects form the canonical inference contract;
- the stability expectations associated with the public API.

Detailed implementation reference documentation may be generated separately.

## API philosophy

BayMorph exposes scientific concepts rather than backend-specific machinery.

The public API is designed around:

- astronomical observations;
- morphology models;
- point-spread functions;
- complete scientific workflows;
- backend-neutral Bayesian inference results.

Scientific meaning has priority over implementation convenience.

Changing the inference backend must not require users to rewrite the scientific
interpretation of their analysis.

## Recommended user workflow

The recommended high-level entry point is:

- `BayMorphRunner`

`BayMorphRunner` coordinates complete BayMorph analyses while preserving the
separation between preparation, scientific model construction, inference, and
reporting.

Users should normally interact with BayMorph through stable high-level
interfaces rather than internal pipeline or fitting classes.

## Scientific data

The public scientific data interfaces are:

- `Observation`
- `ObservationMetadata`
- `ImageData`

These objects represent astronomical data and metadata independently of the
inference backend.

## Morphology models

The current public morphology interfaces are:

- `SersicModel`
- `DoubleSersicModel`
- `CompositeModel`
- `ConvolvedModel`

These objects describe scientific morphology and rendering structure rather
than sampler implementation details.

## Point-spread functions

The current public PSF interfaces are:

- `GaussianPSF`
- `EmpiricalPSF`

PSF objects belong to the observation and rendering model and remain
independent of the selected inference backend.

## Simulation interfaces

The current public simulation interfaces are:

- `SyntheticObservation`
- `SyntheticSersic`

These provide controlled synthetic data products for examples, validation, and
scientific testing.

## Canonical inference results

Production inference backends return a backend-neutral result hierarchy.

The public interfaces are:

- `InferenceResult`
- `PosteriorSamples`
- `EvidenceEstimate`
- `RealFitProducts`

### InferenceResult

`InferenceResult` is the canonical BayMorph inference result.

It separates:

- posterior samples and normalized weights;
- parameter names;
- optional sample log-likelihoods and log-priors;
- Bayesian evidence when available;
- backend diagnostics;
- inference provenance.

Sampler-specific implementation state is intentionally excluded from the
scientific result contract.

For example, a backend is not required to expose nested-sampling-specific
quantities merely because another backend produces them.

### PosteriorSamples

`PosteriorSamples` contains the backend-neutral weighted posterior
representation.

Its responsibilities include:

- preserving sample ordering;
- preserving parameter names;
- normalized posterior weights;
- posterior summary statistics;
- maximum-likelihood and posterior-median parameter access;
- effective sample-size information.


Weighted posterior samples are the authoritative statistical representation.
Posterior summaries and scientific inference operate directly on their
normalized weights.

`equal_weight_samples` provides a deterministic derived representation for
visualization and algorithms that explicitly require unweighted samples.
BayMorph constructs it with fixed-offset systematic resampling of the weighted
empirical posterior. The transformation preserves the number of stored samples
and uses no hidden random-number generator.

Equal-weight samples are therefore a diagnostic convenience rather than a
replacement for the original weighted posterior.

### EvidenceEstimate

`EvidenceEstimate` represents a Bayesian evidence estimate when the selected
backend provides one.

It preserves:

- the logarithmic evidence;
- an optional uncertainty estimate;
- the evidence-estimation method;
- uncertainty semantics;
- diagnostics;
- provenance.

Evidence uncertainty is not assumed to have identical statistical meaning
across inference algorithms.

### RealFitProducts

`RealFitProducts` groups the scientific outputs of a morphology fit, including:

- the canonical `InferenceResult`;
- best-fitting parameters;
- model image;
- residual products;
- goodness-of-fit statistics;
- diagnostic data.

## Inference backend neutrality

Inference backends are implementation mechanisms rather than public scientific
result types.

BayMorph currently supports:

- pocoMC as the default production backend;
- Dynesty static nested sampling;
- Dynesty dynamic nested sampling.

Backend-specific fitter and adapter classes are not part of the stable
top-level API.

Sampler selection and numerical configuration are performed through the
pipeline configuration rather than through backend-specific public workflows.

## Sampler configuration

The public sampler configuration contract is intentionally minimal.

pocoMC uses:

    sampler:
      name: pocomc
      init:
        # Native pocomc.Sampler keyword arguments.
      run:
        # Native pocomc.Sampler.run keyword arguments.

Dynesty static nested sampling uses:

    sampler:
      name: dynesty
      mode: static
      init:
        # Native dynesty.NestedSampler keyword arguments.
      run:
        # Native dynesty.NestedSampler.run_nested keyword arguments.

Dynesty dynamic nested sampling uses:

    sampler:
      name: dynesty
      mode: dynamic
      init:
        # Native dynesty.DynamicNestedSampler keyword arguments.
      run:
        # Native dynesty.DynamicNestedSampler.run_nested keyword arguments.

BayMorph deliberately does not duplicate the complete upstream sampler API.

Arguments under `init` and `run` retain their native upstream names and
semantics. Sampler-specific numerical options should therefore be taken from
the documentation corresponding to the installed pocoMC or Dynesty version.

## Configuration builders

BayMorph exposes three public helpers for constructing reproducible analysis
configurations without requiring users to assemble internal pipeline objects.

### `make_config_from_fits`

`make_config_from_fits` creates a runnable single-observation configuration
from one FITS image.

The public scientific controls include:

- an explicit target-identification seed;
- `model_name`, selecting the registered morphology model;
- `sampler`, accepting either a simple sampler name such as `"pocomc"` or
  `"dynesty"`, or an advanced sampler configuration mapping;
- cutout and PSF strategy;
- output location.

The Config Maker may use deterministic FITS metadata where appropriate, but it
does not infer morphology priors from image intensities.

### `make_multi_observation_config`

`make_multi_observation_config` creates one configuration for each observation
plus the joint configuration required for correlated multi-observation
inference.

Observation identifiers are opaque user-defined labels. BayMorph does not
infer filters, instruments, epochs, or parameter coupling from their names.

The same `model_name` and `sampler` contract used by the single-observation
Config Maker applies to the joint workflow.

### `make_multi_parameters`

`make_multi_parameters` constructs the parameter specification for a registered
morphology model across multiple observations.

Parameter structure is obtained from the model registry. Shared,
observation-specific, and fixed parameters remain explicit modelling choices.

Explicit prior bounds are required where they cannot be defined safely from
the scientific model and deterministic observation metadata. Data-scaled
morphology priors are never generated silently.

## Public top-level namespace

The stable top-level public namespace currently contains:

- `BayMorphRunner`
- `make_config_from_fits`
- `make_multi_observation_config`
- `make_multi_parameters`
- `Observation`
- `ObservationMetadata`
- `ImageData`
- `SersicModel`
- `DoubleSersicModel`
- `CompositeModel`
- `ConvolvedModel`
- `GaussianPSF`
- `EmpiricalPSF`
- `SyntheticObservation`
- `SyntheticSersic`
- `PosteriorSamples`
- `EvidenceEstimate`
- `InferenceResult`
- `RealFitProducts`

These names are available directly from `baymorph`.

## Legacy result compatibility

`PosteriorResult` is the historical BayMorph result representation.

It is retained within `baymorph.results` for compatibility with legacy result
archives and internal migration paths.

It is not part of the stable top-level public API.

New user code and documentation should use `InferenceResult`,
`PosteriorSamples`, and `EvidenceEstimate`.

Legacy persisted results may be migrated into the canonical
`InferenceResult` representation when loaded.

## Internal APIs

Implementation modules such as the following are not stable public interfaces:

- `baymorph.fitting`
- `baymorph.pipeline`
- `baymorph.rendering`
- `baymorph.validation`

This includes backend adapters and backend-specific fitter classes such as
Dynesty and pocoMC implementation objects.

These modules may evolve as the architecture develops.

Users writing scientific analyses should prefer the stable top-level API and
configuration-driven workflows.

## Extension points

BayMorph supports extension through composition.

Relevant extension points include:

- morphology models;
- observation operators;
- likelihoods;
- prior models;
- inference backends;
- diagnostics;
- validation experiments.

New abstractions should be introduced only when multiple concrete scientific
use cases justify them.

Extensions must preserve the separation between:

- scientific model;
- numerical backend;
- backend-neutral scientific result.

## API stability policy

BayMorph distinguishes four interface states.

| Status | Meaning |
| --- | --- |
| **Stable** | Backward compatibility is expected across minor releases. |
| **Experimental** | The interface may evolve as the scientific architecture matures. |
| **Internal** | No backward-compatibility guarantee is provided. |
| **Deprecated** | The interface is retained temporarily during migration. |

Scientific correctness takes precedence over preserving an incorrect or
misleading interface.

## Compatibility policy

Backward compatibility should be preserved for stable public interfaces when
scientifically reasonable.

Compatibility must not preserve scientifically incorrect behaviour.

When a scientific or architectural correction requires an API change, the
transition should be:

1. explicit;
2. documented;
3. tested;
4. accompanied by a migration path where practical.

## Relationship to the architecture

The public API implements the scientific contracts defined in:

- `docs/architecture/CORE_ARCHITECTURE.md`
- `docs/architecture/OBSERVATION_MODEL.md`
- `docs/architecture/INFERENCE.md`
- `docs/architecture/RENDERING.md`
- `docs/architecture/PIPELINE.md`
- `docs/architecture/VALIDATION.md`

The architecture defines responsibilities.

The public API exposes the subset of those responsibilities that should remain
stable for scientific users.
