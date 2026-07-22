# Changelog

All notable changes to BayMorph will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project follows semantic versioning in spirit.

## [Unreleased]

### Added

### Changed

### Deprecated

### Removed

### Fixed

### Security

## [0.2.0] - 2026-07-22

### Added

- Layered package architecture with dedicated core, model, fitting, pipeline, plotting, simulation, results, and runner layers.
- `Observation`, `ImageData`, and observation-metadata abstractions.
- High-level `BayMorphRunner` orchestration.
- Configuration-driven simulation, image-preparation, PSF, and fitting workflows.
- Single- and double-Sérsic model infrastructure with model-agnostic rendering interfaces.
- Dynesty nested-sampling backend and evidence-aware model-comparison infrastructure.
- Automatic radial-profile plateau detection and adaptive target cutouts.
- Target-specific sky-region construction, background estimation, and noise-map generation.
- Empirical PSF candidate selection, construction, validation, persistence, and loading workflows.
- Reusable result products, diagnostics, posterior summaries, and plotting utilities.
- Command-line project initialization and packaged example installation.
- Expanded automated tests and architecture, validation, roadmap, and decision documentation.

### Changed

- Consolidated scientific workflows into reusable package modules instead of notebook-local implementations.
- Standardized runtime configuration resolution and persisted resolved configurations.
- Established explicit output-tree and product-loading conventions.

### Fixed

- Improved masking, source-selection, cutout-boundary, PSF-normalization, and background-estimation robustness across real-image workflows.

[Unreleased]: https://github.com/carlocannarozzophd/BayMorph/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/carlocannarozzophd/BayMorph/releases/tag/v0.2.0
