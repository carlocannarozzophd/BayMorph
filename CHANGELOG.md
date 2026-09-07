# Changelog

All notable changes to BayMorph are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project follows Semantic Versioning in spirit.

---

## [Unreleased]

### Added

* Added prior-volume coverage review for observationally calibrated flux priors and weighted post-fit diagnostics for posterior contact with declared prior boundaries.
* Canonical multi-observation configuration, preparation, joint-inference,
  persistence, and diagnostic workflows.
* Explicit shared, observation-specific, fixed, deterministic, and
  hierarchical parameter-coupling infrastructure.
* Native-grid and WCS-aware modelling of heterogeneous observations.
* Nautilus production backend alongside Dynesty and pocoMC.
* Within-observation correlated-noise calibration and correlated Gaussian
  image likelihoods.
* Likelihood normalization-status and per-observation likelihood provenance.
* Hybrid Sérsic rendering with validated adaptive numerical refinement.
* Observation Advisor, guided configuration, and evidence-aware flux-prior
  calibration workflows.
* Experimental preparation-free model screening with simulation generation,
  interpretable features, classifier training, evaluation, calibration, safe
  persistence, and mandatory user model selection.
* Backend-neutral validation runners, reference problems, repeatability
  analyses, and simulation-based calibration infrastructure.
* Canonical simulated single-observation, observed single-observation, and
  multi-observation public notebooks.
* Packaged generic tutorial notebooks copied into new user workspaces by
  `baymorph init`.
* Packaging tests for runtime, development, and interactive dependencies.
* Public-source boundary checks preventing observational data and generated
  research products from entering release artifacts.

### Changed

* Replaced legacy multiband terminology and runtime objects with the general
  multi-observation architecture.
* Made inference backends consume only backend-neutral `InferenceProblem`
  instances.
* Consolidated posterior samples, Bayesian evidence, diagnostics, timing, and
  provenance in backend-neutral result objects.
* Separated inference footprints from preparation footprints.
* Standardized typed prior configuration and parameter routing.
* Made plotting independent of an external LaTeX installation.
* Separated model-screening runtime code from validation and training
  infrastructure.
* Consolidated architecture, API, workflow, validation, and research
  documentation around the production contracts.
* Refocused the roadmap on real multi-observation production fits and paper
  closure.
* Generalized the observed single- and multi-observation notebooks for
  user-supplied FITS data, explicit target seeds, and observation-specific
  flux priors.
* Adopted normalized Gaussian likelihoods as neutral tutorial defaults while
  requiring explicit validation for correlated-noise models.

### Removed

* Experimental native nested-sampler runtime and its production integration.
* Superseded geometry-specific sampler infrastructure.
* Legacy multiband compatibility modules and obsolete configuration paths.
* Obsolete validation backends, notebooks, audit artifacts, and broken
  benchmarks.
* Target-specific public configuration and tracked generated benchmark tables,
  research summaries, and figures.

### Fixed

* Resolved per-observation likelihood overrides with explicit precedence over
  joint defaults.
* Ensured correlated-noise calibration is constructed before joint inference
  for every observation that requires it.
* Made multi-observation fit statistics use each observation's actual
  likelihood and effective noise state.
* Persisted per-observation likelihood metadata and normalization status in
  joint products.
* Corrected likelihood-consistent Mahalanobis statistics for correlated
  observations.
* Declared all required runtime dependencies and separated optional
  interactive and development dependencies.
* Improved PSF selection, mask handling, background estimation, radial
  diagnostics, output persistence, and sampler provenance.
* Corrected public notebook, API, terminology, and release documentation
  inconsistencies.

---

## [0.3.0] - 2026-07-23

### Added

* Stable layered architecture separating core, models, fitting, pipeline,
  plotting, simulation, results, and high-level orchestration.
* High-level `BayMorphRunner` interface for end-to-end workflows.
* `Observation`, `ImageData`, and observation metadata abstractions.
* Configuration-driven workflows for simulation, image preparation, PSF
  handling, Bayesian fitting, and diagnostics.
* Modular single- and double-Sérsic model infrastructure with model-agnostic
  rendering interfaces.
* Bayesian inference based on Dynesty nested sampling.
* Bayesian evidence infrastructure for objective model comparison.
* Automatic plateau-based cutout determination.
* Target-specific sky-region construction with empirical background and noise
  estimation.
* Empirical PSF selection, construction, validation, persistence, and loading
  workflows.
* Comprehensive diagnostic products, posterior summaries, and plotting
  utilities.
* Command-line utilities for project initialization and packaged example
  installation.
* Complete project documentation, including API reference, contributing guide,
  roadmap, architectural decisions, and citation metadata.
* Extensive automated test suite covering scientific workflows, packaging,
  command-line interfaces, and resource consistency.

### Changed

* Consolidated notebook implementations into reusable package modules.
* Standardized configuration loading, validation, resolution, and persistence.
* Standardized output directory structure and generated scientific products.
* Unified model, likelihood, and sampler abstractions to simplify future
  backend extensions.

### Fixed

* Improved robustness of source detection, masking, cutout generation,
  background estimation, PSF normalization, and real-image preparation
  workflows.
* Resolved multiple packaging and resource-distribution inconsistencies.
* Improved reliability of diagnostic generation and posterior summary
  products.

---

[Unreleased]: https://github.com/carlocannarozzophd/BayMorph/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/carlocannarozzophd/BayMorph/releases/tag/v0.3.0
