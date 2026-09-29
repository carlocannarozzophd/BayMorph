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

### Repeating a stage

`prepare_image()`, `select_psf_candidates()`, `build_psf()` and `fit()` can be
run again. A stage that finds a completed product for the configured output
namespace loads it, and reports the reuse, instead of rebuilding it (AD-042).
An interrupted workflow therefore continues by rerunning it from the start.

Reuse is the fail-closed path, not a shortcut: persisted preparation products
are accepted only when their immutable inputs still match the configuration,
and an empirical PSF only when its kernel and provenance validate. Products
that no longer match are rejected, and none are ever rewritten in place.
Setting `outputs.overwrite` selects rebuilding; a different scientific
declaration belongs in a new output namespace.

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
- `make_observation`

`make_observation` is the high-level constructor for controlled synthetic
astronomical observations. These interfaces support examples, validation, and
scientific testing without exposing simulation implementation details.

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

`InferenceResult.equal_weight_samples` exposes deterministic equal-weight
samples for visualization. The weighted samples remain the statistical source
of truth for every summary, credible interval, and evidence quantity.

#### Posterior calibration

`InferenceResult.mahalanobis_distance(reference)` measures how far a reference
parameter vector, typically a simulation truth, lies from the posterior while
accounting for parameter correlations. It returns the parameters used, the
squared distance, the distance, the degrees of freedom, and a chi-squared
p-value.

Correlated parameters must be assessed jointly. Reading marginal
one-parameter offsets of a correlated posterior double-counts a single shared
displacement and overstates the tension; Sérsic `flux`, `re`, and `n` are
strongly correlated in practice.

Three limits apply:

- the metric uses the weighted samples, not derived equal-weight samples;
- the p-value assumes an approximately Gaussian posterior. The distance stays
  defined for a non-Gaussian posterior, but its p-value does not;
- a single realization does not test calibration. Separating an unlucky noise
  realization from a genuine bias requires repeated experiments, which is what
  `validation/repeated_noise_recovery/` provides.

A singular covariance raises rather than falling back to a pseudo-inverse,
because a pseudo-inverted metric returns a plausible number that is not the
declared quantity.

The underlying `weighted_covariance` and `mahalanobis_distance_squared`
functions are available from `baymorph.stats`.

#### Prepared noise scale

The prepared noise field of the canonical single-observation workflow is a
truncation-corrected Gaussian scale: sky pixels are iteratively sigma-clipped,
the standard deviation of the survivors is computed, and the truncation caused
by the clipping is inverted to recover the parent scale. See AD-040.

The estimator it replaced, a scaled median absolute deviation, was not
materially biased; it was inefficient. A scaled MAD has roughly 37% efficiency,
so its run-to-run scatter was about 1.1% at the sky-region sample size of the
canonical example, against 0.7% now. That scatter enters the Gaussian
likelihood normalization, so it sets every credible-interval width and
displaces absolute `logZ` by order 100.

Clipping is retained: the correction restores efficiency, not robustness.

The scale is measured from residuals about a **plane** fitted to the sky
region, not about a constant, so an unmodelled sky gradient is no longer
counted as noise. See AD-041. Preparation still subtracts a constant:
the plane only measures the scale.

A consequence worth knowing before it surprises you: because the gradient is
no longer absorbed into the noise, an unmodelled gradient now surfaces as an
**elevated reduced chi-squared** instead of being hidden. That is deliberate.
The remedy is `fit.background.model: plane`, which adds
`background_gradient_x` and `background_gradient_y` as fitted parameters.

Preparation reports the fitted gradient, its standard error and its
significance per axis, in the preparation metadata and in the runtime report
under `image_preparation.background.sky_gradient`. It is a diagnostic: a
gradient significant at several sigma indicates that a plane background is
warranted, and the choice remains the user's. BayMorph ranks; it does not
select.

Three limitations remain, all explicit:

- the correction assumes the surviving core is Gaussian;
- only a linear gradient is removed, so higher-order sky structure is still
  counted as noise;
- the prepared noise field excludes source Poisson variance and any external
  variance map.

`baymorph.stats.truncated_gaussian_sigma` and `truncated_variance_ratio`
expose the correction directly.

The separate `estimate_noise_map` serves the observation advisor and
empirical-PSF candidate selection. It does not set the likelihood noise.

#### Corner-plot truth markers

`baymorph.plotting.corner_plot` accepts `truths`, a sequence ordered as
`parameter_labels`, together with `truth_color` and `truth_linewidth`. Entries
that are `None` or NaN leave their parameter unmarked.

Truth markers are a visualization overlay. They do not enter any posterior
summary, credible interval, median marker, or Bayesian evidence value. Axis
limits follow the posterior samples and are not expanded, so a truth outside
the plotted range is clipped rather than shown.

Corner diagnostics add these markers automatically for single-observation runs
whose synthetic observation was generated by the pipeline, detected by the
presence of the persisted `truth.json`. Real-data fits therefore never receive
truth markers. Two transformations are applied when building the vector:

- truth `x0`/`y0` are full-image coordinates and are shifted by the prepared
  cutout origin, because sampled positions are cutout coordinates;
- the simulated `sky` is never mapped onto `background_level`. The sky plateau
  is removed during image preparation, so the sampled background is a residual
  of the prepared image and not the simulated sky.

Multi-observation corner diagnostics remain unmarked: each observation has its
own cutout origin and there is no single joint simulation truth.

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

- Nautilus as the default production backend;
- Dynesty static nested sampling;
- Dynesty dynamic nested sampling;
- pocoMC.

Backend-specific fitter and adapter classes are not part of the stable
top-level API.

Sampler selection and numerical configuration are performed through the
pipeline configuration rather than through backend-specific public workflows.

## Sampler configuration

The public sampler configuration contract is intentionally minimal.

Nautilus, the default production backend, uses:

    sampler:
      name: nautilus
      init:
        # Native nautilus.Sampler keyword arguments, for example n_live,
        # seed, filepath, resume.
      run:
        # Native nautilus.Sampler.run keyword arguments, for example n_eff.

emcee uses a convergence-driven run loop rather than a thin passthrough,
because emcee has no native evidence-producing stopping rule:

    sampler:
      name: emcee
      init:
        nwalkers: auto          # 'auto' or a positive even integer
        seed: 42                # optional
        checkpoint_path: auto   # optional; 'auto' uses the managed run directory
        checkpoint_mode: fresh  # 'fresh' or 'resume'
      run:
        burn_in: 500
        max_steps: 10000
        check_interval: 100
        min_autocorr_times: 50.0
        tau_tolerance: 0.05

emcee is posterior-only: it does not produce a Bayesian evidence estimate and
cannot be used when `fit.comparison.enabled` is true. `init.checkpoint_path`,
`init.checkpoint_mode`, and `init.checkpoint_signature` are BayMorph-owned
options layered on top of the native emcee constructor arguments; they are
not native `emcee.EnsembleSampler` keywords.

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

### Checkpoint and resume safety

For Nautilus (`init.filepath`) and pocoMC (`run.resume_state_path`), BayMorph
requires a sibling `<checkpoint>.target.json` signature recording the exact
scientific problem, sampler settings, and BayMorph source that produced it.
Both accept `auto`, resolving to a path inside the pipeline-managed run
directory. Resuming from a checkpoint whose recorded signature does not match
the current problem raises `RuntimeError`; a checkpoint with no recorded
signature at all is never trusted. For emcee, the equivalent contract is the
embedded `init.checkpoint_signature` described above. This applies to both
single-observation and joint multi-observation fits.

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
- `background`, an explicit residual-background declaration naming the model
  (`constant` or `plane`) and a typed prior for exactly the parameters that
  model samples, in science-image units;
- `psf_max_peak`, an explicit ceiling on the peak pixel of empirical-PSF
  candidate stars, in science-image units;
- output location.

The Config Maker may use deterministic FITS metadata where appropriate, but it
does not infer morphology priors from image intensities. It never converts a
FITS saturation keyword into `psf_max_peak`, because the keyword's units are
not guaranteed to match the image. When `background` or `psf_max_peak` is
omitted, the resolved configuration falls back on BayMorph defaults that are
not scaled to the image units.

### `make_multi_observation_config`

`make_multi_observation_config` creates one configuration for each observation
plus the joint configuration required for multi-observation inference.

Observation identifiers are opaque user-defined labels. BayMorph does not
infer filters, instruments, epochs, or parameter coupling from their names.

The same `model_name` and `sampler` contract used by the single-observation
Config Maker applies to the joint workflow. The `background_model` argument
declares the residual-background family, `constant` or `plane`, recorded in
the joint `fit.background` section; the per-observation background priors are
part of `parameters` and must use independent coupling.

### `make_multi_parameters`

`make_multi_parameters` constructs the parameter specification for a registered
morphology model across multiple observations.

Parameter structure is obtained from the model registry. Shared,
observation-specific, and fixed parameters remain explicit modelling choices.
The residual-background parameters selected by `background_model` are
appended to the morphology contract and are always observation-specific.

Explicit prior bounds are required where they cannot be defined safely from
the scientific model and deterministic observation metadata. Data-scaled
morphology priors are never generated silently.

## Observation advisor

`ObservationAdvisor` inspects one FITS image and builds a reviewable,
evidence-aware analysis configuration. `propose_config` never writes a file;
it returns an `ObservationConfigProposal` recording the proposed
configuration, the starting model, the flux-prior source, whether the flux
prior requires human review, and any warnings. The proposal must be reviewed
and explicitly accepted with `write(accept=True)` before anything is written
to disk. Writing the same proposal again changes nothing, so a notebook can
be re-executed, but `write` refuses to replace a file whose content differs
unless `overwrite=True` is also passed; replacing a configuration does not
update products already created from it.

The proposal records every prior the fit will sample: the complete prior set
of the starting model and the priors of the parameters sampled by the
residual-background model. Priors that were not declared are BayMorph
reference defaults; the proposal writes them explicitly and names them in a
warning, so the YAML does not depend on defaults that a later version could
change. It also warns when no residual-background prior or no empirical-PSF
candidate peak ceiling was declared, without choosing either value. The Advisor
summary reports the FITS `BUNIT` and saturation keyword as header metadata
that is not unit-checked, and labels its background and noise scales as
whole-image diagnostics: the noise scale used by the likelihood is measured
on the prepared cutout.

## Model screening advisor

`SingleObservationModelAdvisor` performs preparation-free screening of one
FITS observation against the implemented morphology-model classes.

The workflow:

- reads normalized observational metadata;
- extracts deterministic, interpretable image features;
- applies a persisted simulation-trained classifier;
- returns a ranked `ModelScreeningReport`;
- always leaves production model selection to the user.

`load_model_screening_classifier` loads a validated
`ModelScreeningClassifierArtifact`. Screening outputs are represented by
`SingleObservationScreeningResult`, `ModelScreeningReport`, and
`ModelScreeningScore`.

Screening scores are conditional on the declared simulation and training
distributions. They are not real-data posterior model probabilities and are
not Bayesian evidences. Screening does not perform production cleaning,
masking, background modelling, PSF construction, or inference.

## Public top-level namespace

The stable top-level public namespace currently contains:

- `BayMorphRunner`
- `ObservationAdvisor`
- `ObservationConfigProposal`
- `ObservationInspection`
- `PriorGuidance`
- `make_config_from_fits`
- `make_observation`
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
- `SingleObservationModelAdvisor`
- `SingleObservationScreeningResult`
- `ModelScreeningReport`
- `ModelScreeningScore`
- `ModelScreeningClassifierArtifact`
- `load_model_screening_classifier`

These names are available directly from `baymorph`.

## Legacy result archive compatibility

Historical BayMorph posterior archives remain loadable through
`InferenceResult.load()`.

Legacy archive fields are interpreted only at the persistence boundary and are
immediately migrated into `PosteriorSamples` and `EvidenceEstimate`.
The historical `PosteriorResult` runtime class is no longer part of BayMorph.

New code should use only `InferenceResult`, `PosteriorSamples`, and
`EvidenceEstimate`.

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
