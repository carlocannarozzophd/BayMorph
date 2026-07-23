# Changelog

All notable changes to BayMorph are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project follows Semantic Versioning in spirit.

---

## [Unreleased]

### Added

### Changed

### Deprecated

### Removed

### Fixed

### Security

---

## [0.2.0] - 2026-07-22

### Added

* Stable layered architecture separating core, models, fitting, pipeline, plotting, simulation, results, and high-level orchestration.
* High-level `BayMorphRunner` interface for end-to-end workflows.
* `Observation`, `ImageData`, and observation metadata abstractions.
* Configuration-driven workflows for simulation, image preparation, PSF handling, Bayesian fitting, and diagnostics.
* Modular single- and double-Sérsic model infrastructure with model-agnostic rendering interfaces.
* Bayesian inference based on Dynesty nested sampling.
* Bayesian evidence infrastructure for objective model comparison.
* Automatic plateau-based cutout determination.
* Target-specific sky-region construction with empirical background and noise estimation.
* Empirical PSF selection, construction, validation, persistence, and loading workflows.
* Comprehensive diagnostic products, posterior summaries, and plotting utilities.
* Command-line utilities for project initialization and packaged example installation.
* Complete project documentation, including API reference, contributing guide, roadmap, architectural decisions, and citation metadata.
* Extensive automated test suite covering scientific workflows, packaging, command-line interface, and resource consistency.

### Changed

* Consolidated notebook implementations into reusable package modules.
* Standardized configuration loading, validation, resolution, and persistence.
* Standardized output directory structure and generated scientific products.
* Unified model, likelihood, and sampler abstractions to simplify future backend extensions.

### Fixed

* Improved robustness of source detection, masking, cutout generation, background estimation, PSF normalization, and real-image preparation workflows.
* Resolved multiple packaging and resource-distribution inconsistencies.
* Improved reliability of diagnostic generation and posterior summary products.

---

[Unreleased]: https://github.com/carlocannarozzophd/BayMorph/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/carlocannarozzophd/BayMorph/releases/tag/v0.2.0
