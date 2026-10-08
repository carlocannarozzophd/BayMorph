# Changelog

All notable changes to BayMorph are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project follows Semantic Versioning in spirit.

---

## [Unreleased]

### Fixed

* A new BayMorph release no longer rejects, by itself, the sky correlation
  of a preparation and its nearest-neighbour factorization. They are
  accepted when the code that computes them and the versions of NumPy and
  SciPy are unchanged, and the BayMorph version that made them stays
  recorded with them. Products of an earlier schema, or computed by other
  code, are still rejected.

---

## [0.5.0] - 2026-10-07

This release completes the observation advisor. It reads the FITS header,
prepares the image and builds its PSF, and proposes every prior of one or
several models from the header, from physics and from those fixed inputs,
never from the image it is about to fit; nothing is written before you
review and accept the proposal. The advisor works from Python, from
`baymorph config --advise` and in the observed-image tutorials.
Configurations written for 0.4.0 run unchanged.

### Added

* `ObservationAdvisor.propose_config` proposes every prior of the
  configuration from the header, from physics and from the fixed inputs of
  the observation model, never from the image it is about to fit: a flux
  from the magnitudes 8 to 30 through the zeropoint of the header, a length
  of a one-component model from 0.1 to 120 arcsec through the pixel scale,
  the centre within three times the FWHM of the PSF, and the residual
  background within five times the sky scale. The lengths of models with
  several components keep the ranges that tell their components apart.
  The advisor also proposes the source of the effective gain, with a
  cross-check of the gain keyword, and the ceiling on the peak of the PSF
  stars from the saturation level when its unit is known. Declared priors
  and settings are kept as they are.
* `ObservationAdvisor.measure_fixed_inputs` prepares the image and builds
  its PSF before the priors are proposed; the fit reuses these products.
* Writing a proposal also writes a JSON report beside the YAML that records
  where every proposed value comes from.
* `make_config_from_fits(preparation=...)` writes a `preparation` section.
* `baymorph.models.labels.PARAMETER_KINDS`: the kind of every parameter of
  the ten models (flux, length, position, angle or dimensionless).
* `ObservationAdvisor.propose_config(models=[...])` proposes one
  configuration for several models. Each model receives its priors by the
  same rules, `model_priors` maps each model to the priors declared for
  it, and the fit runs one fit per model, each with its own evidence.
* `ObservationAdvisor.header_report()` lists what the header holds: each
  quantity with its value, unit, status and source.
* `ObservationAdvisor.prior_guidance` covers the ten models.
* `baymorph config --advise` prints the header report and the proposal of
  the advisor, and writes the configuration and its report only with
  `--accept`. With `--advise`, `--model` may be repeated, and
  `--background-model` and `--resampling-variance-factor` reach the
  advisor.
* `baymorph config --hdu`, `--psf-max-peak` and `--sky-correlation`.
* Tutorial 02A guides the advisor from the header to the reviewed
  configuration, and prepares the image and the PSF; tutorial 02B reuses
  them and shows the posterior, residuals and evidence of each model.

### Changed

* `ObservationAdvisor.propose_config` no longer requires a flux prior when
  the header gives a zeropoint, and it prepares the image and builds the
  PSF unless `fixed_inputs=` is supplied.
* Generated configurations no longer write the unused key
  `psf.empirical.candidate_selection.auto`.
* `baymorph config --uniform-prior` accepts a parameter written as
  `MODEL.PARAMETER`, which it requires when several models are chosen.

### Fixed

* `ObservationAdvisor.measure_fixed_inputs` sets the ceiling on the peak of
  the PSF stars from the saturation level of the header, as
  `propose_config` does. Fixed inputs measured first are therefore accepted
  by a proposal with the same options.
* The advisor proposes the axis ratios and position angles of the two
  components of `double_sersic` from the defaults the fit uses, instead of
  refusing to propose them.

---

## [0.4.0] - 2026-10-06

This release changes how BayMorph models the noise, the background and the
PSF of a real image, adds fits of several observations of one object and
six morphology models, and rewrites the documentation. Configurations and
products of 0.3.0 must be updated: see "Upgrading from 0.3.0" below.

### Added

* A noise model for real images: a normalized Gaussian likelihood whose
  variance in each pixel is the sky variance plus the Poisson variance of
  the source, `s^2 + max(m, 0) / g`, with the correlation of the sky
  measured on the sky region of the preparation and a nearest-neighbour
  factorization of 24 neighbours. The effective gain `g` is declared in
  `fit.likelihood.effective_gain` as `declared`, `sky_level`,
  `measured_sky` or `absent`; it is never fitted and has no default.
* `preparation.noise.sky_correlation: field_separated`, an estimator of the
  sky correlation for skies that hold a large-scale field. It is declared
  in the configuration, never chosen automatically.
* Effective-PSF rendering: an empirical PSF carries its kernels at
  sub-pixel offsets, and models are convolved with them instead of being
  sampled at pixel centres. The empirical-PSF builder measures the stars at
  each sub-pixel phase.
* Six morphology models for single-observation fits, beside
  `single_sersic`, `double_sersic`, `sersic_point_source` and `bulge_disk`:
  `boxy_sersic`, `core_sersic`, `bulge_broken_disk`, `bulge_disk_bar`,
  `bulge_disk_ring` and `bulge_disk_spiral`. Configuration validation
  rejects priors outside the domain in which their rendering is validated,
  which the page of each model states.
* Fits of several observations of one object: one configuration, the
  preparation of each observation on its own grid with its own PSF, mask
  and noise, joint inference and joint products, with explicit coupling of
  the parameters across observations (shared, per observation, fixed,
  hierarchical or wavelength-dependent).
* A residual background fitted with the galaxy, `fit.background`, constant
  or a plane, with its priors under `fit.background.priors`.
* Nautilus, the default sampler, and pocoMC, beside Dynesty.
* The header of a FITS image is read through conventions: the FITS
  standard, ESO products, SWarp and VST-Tube coadds, SDSS, the Legacy
  Surveys, HST, JWST, DECam, hscPipe, Pan-STARRS and astrocut cutouts.
  `data.hdu` selects an image stored in an extension.
* The summary of a preparation records the fraction of the detected target
  that the fitting mask excludes.
* `InferenceResult.equal_weight_samples` for plots, beside the weighted
  samples that remain the statistical result, and
  `InferenceResult.mahalanobis_distance()`.
* Truth markers in the corner plots of simulated fits, and a diagnostic of
  posteriors that reach the bounds of their priors.
* Tutorial notebooks for a simulated galaxy and for an observed galaxy, in
  two parts: its assessment and configuration, and its fit. `baymorph init`
  copies them.
* A documentation site written for users: a User Guide with one page for
  each stage of an analysis, a Theory section, a page for each model and
  component with its figures and references, and a Reference section for
  the command line, the configuration, the output files and the Python API.

### Changed

* Every fit run from a configuration uses the noise model above and
  requires a declared effective gain. `baymorph config` declares
  `effective_gain: {source: measured_sky}` only when the sky of the image
  allows it, and says so otherwise.
* The preparation masks a detected segment only when it holds a source
  above the light of the target (`preparation.masking.contaminant_rule:
  local_excess`, the default; `all_segments` keeps the previous masks),
  grows the masked segments by 20 px for the sky region and the likelihood
  support (`contaminant_growth_pixels`, 0 for a curated mask used as it
  is), and excludes pixels more than 5 sky scales below zero
  (`negative_threshold_sigma`). On simulated images of a galaxy among
  neighbours, the light of the neighbours beyond their detected segments
  biased the flux, the effective radius and the Sérsic index by up to 5
  percent; the grown masks keep these biases within 0.2 percent on a flat
  sky.
* The noise scale of the preparation is a Gaussian scale corrected for the
  truncation and measured about a plane fitted to the sky region.
* FITS images whose pixel values are not linear in flux, such as
  Pan-STARRS stacks compressed with `asinh`, are refused.
* `DEFAULT_CONFIG` proposes `single_sersic` instead of `double_sersic`.
* Faster fits: the PSF convolution keeps the transform of its kernel, and
  Nautilus installs the problem in each worker once instead of sending it
  with every task.
* BayMorph depends on Nautilus, pocoMC, h5py, multiprocess and
  threadpoolctl, besides Dynesty; the interactive and development tools
  are optional extras.
* The documentation site is built with Zensical.

### Removed

* The independent Gaussian likelihood for fits run from a configuration; it
  remains in the library for simulations and tests.
* The model parameter `sky`: the residual background is `fit.background` in
  a single fit and a parameter of `fit.multi_observation.parameters` in a
  joint fit.
* `data.cutout_size`: a fixed cutout is declared as `data.cutout` with
  `mode: fixed` and `size`.

### Fixed

* Source detection no longer falls back silently to a background that
  ignores the detection mask.
* Fixes in the preparation, the selection of PSF stars, the masks, the
  estimation of the background, the diagnostics and the saved products.

### Upgrading from 0.3.0

* Declare `fit.likelihood.effective_gain` in every configuration.
* Remove the `sky` parameter, and give the background priors under
  `fit.background.priors`, or under `fit.multi_observation.parameters` in
  a joint fit.
* Replace `data.cutout_size` by `data.cutout` with `mode: fixed` and
  `size`.
* Prepare the images and build the PSFs again in new output namespaces:
  the preparation products, PSFs and checkpoints of 0.3.0 are rejected,
  never converted.
* Remove `data.noise_map_path` and `data.psf_path`, even when they are
  null: external noise maps and PSF files are not supported yet.

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

[Unreleased]: https://github.com/carlocannarozzophd/BayMorph/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/carlocannarozzophd/BayMorph/releases/tag/v0.5.0
[0.4.0]: https://github.com/carlocannarozzophd/BayMorph/releases/tag/v0.4.0
[0.3.0]: https://github.com/carlocannarozzophd/BayMorph/releases/tag/v0.3.0
