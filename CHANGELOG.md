# Changelog

All notable changes to BayMorph are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project follows Semantic Versioning in spirit.

---

## [Unreleased]

### Added

* `data.calibration` declares the calibration quantities that the header
  does not give, or gives wrongly: the image unit, the zeropoint and its
  photometric system, the exposure time, the gain of the detector and the
  saturation level. A declared value takes the place of the header reading
  with the status `declared`, the header reading stays recorded beside it,
  and a declared unit reaches the quantities that depend on it, such as the
  zeropoint of an HST image. Declarations change no prepared product. A
  joint configuration does not accept them yet.
* `baymorph config` declares them with the options `--image-unit`,
  `--zeropoint`, `--photometric-system`, `--exposure-time`, `--gain` and
  `--saturation`, with or without `--advise`, and `make_config_from_fits`
  with its argument `calibration`.
* `ObservationAdvisor(..., calibration=...)` and
  `ObservationAdvisor.from_project(..., calibration=...)` take the same
  declarations. The header report, the inspection and the proposal read the
  header with them, and the proposed configuration and its report carry them,
  each with its source.
* Without a usable zeropoint, the advisor proposes every flux prior instead
  of stopping: log-uniform in image units, from the flux of a point source at
  signal-to-noise 1 on the prepared sky, the sky scale times the square root
  of the noise-equivalent area of the PSF, to 22 magnitudes above it. The
  noise-equivalent area of the PSF is a new fixed input
  (`FixedInputs.psf_noise_equivalent_area`).
* Two header conventions. HST images drizzled to counts per second
  (`D001OUUN = 'cps'`) from exposures in electrons are read in electrons
  per second even when their `BUNIT` says `ELECTRONS`, and their AB
  zeropoint is derived from `PHOTFLAM` and `PHOTPLAM` whatever their
  `FILENAME`. The mosaics of grizli, such as those of the DAWN JWST Archive,
  take their AB zeropoint from `PHOTFNU`, 28.9 for pixels of 10 nJy. The
  filter of an ACS image is read from `FILTER1` and `FILTER2`, leaving out a
  `CLEAR` element.
* Two more header conventions, for High Level Science Products of MAST that
  document their zeropoints by filter: the Hubble eXtreme Deep Field
  (`XDFVER`; for example 24.87 AB in F850LP) and the HST/ACS Coma Cluster
  Treasury Survey (25.937 AB in F814W, 26.068 in F475W). The XDF headers
  carry no `PHOTFLAM`; the Coma headers record `PHOTCORR = OMIT`, some with a
  `PHOTFLAM` of 0, and where `PHOTFLAM` is valid it agrees with the survey
  zeropoint within 0.001 mag, and the survey value is kept. A convention may
  now tabulate a quantity by a text quantity read before, such as a zeropoint
  by filter (`by` and `table`), and a reading that overrides another records
  the overridden value in its note.
* The observation advisor asks an online calibration service for a
  calibration quantity that the header lacks and that is not declared, and
  writes the answer to `data.calibration`, so the fit still reads it from the
  YAML alone, offline. The first service is the ACS Zeropoint Calculator of
  STScI: for an ACS image of the WFC or HRC detector drizzled to electrons
  per second whose header gives no usable zeropoint, and not two readings of
  it that disagree, the advisor asks once, over HTTPS and following no
  redirection, for the AB zeropoint of its detector and filter on the date of
  the middle of the observation, between `EXPSTART` and `EXPEND`, accepts the
  answer only when its AB and ST zeropoints agree with its `PHOTFLAM` and
  pivot wavelength, and declares it with `photometric_system: AB`. For an
  observation over several dates it also asks for its first and last dates,
  and declares nothing when their zeropoints differ from that of the middle
  date by more than 0.002 mag, as across the step of July 2006 of WFC. It does
  not ask for an image without observation dates, such as a mosaic of
  several epochs, nor for dates that span more than a year or do not cover
  the exposure time. A zeropoint read from the header or declared is never
  replaced. The header report, the summary and the report of the proposal
  state what was asked, the answer and when it was asked, or why the
  service was not asked; without a usable answer the flux priors follow the
  rule without a zeropoint, so the evidences of proposals made with and
  without it are not comparable. `ObservationAdvisor(..., online=False)` and
  `baymorph config --advise --offline` ask no service.
* The header reading has three new quantities, `detector`,
  `observation_start` and `observation_end`, which HST images give in
  `DETECTOR`, `EXPSTART` and `EXPEND`, the last two as Modified Julian Dates.
* `data.region` reads a large image within a working region about the
  target. By default an image with a side larger than 4096 pixels is read
  within a square of 4096 pixels about `data.target_position`, about 134 MB
  in double precision whatever the size of the mosaic; `whole` reads the
  whole image, and `{size: S}` a square of S arcsec. Only the pixels of the
  region are read: the bytes that hold them in an uncompressed file and the
  tiles that cross it in a tile-compressed one, while a file compressed as a
  whole, gzip or bzip2, is still read through up to the region, in one pass.
  Every position stays in pixels of the whole image: the target, the cutout
  bounds, the PSF stars, and the axes of the first panel of the fit report.
  A cutout, or a target of an automatic cutout, that leaves the region stops
  with the side to declare. A fixed cutout inside the region gives the same
  prepared products as the whole image. The region enters the identity of
  the prepared products and of the PSF only when it is not the whole image,
  so products of images read whole keep their identity.
* `baymorph config` declares the region with `--region-size` or
  `--whole-image` and says how much of the image will be read;
  `make_config_from_fits(..., region=...)`,
  `ObservationAdvisor(..., region=...)` and
  `ObservationAdvisor.from_project(..., region=...)` take the same values.

### Changed

* When several header conventions read the same quantity, the value of the
  one that decides is compared with the others, and two values that disagree
  make the quantity `conflicting`, with both sources in its note, unless the
  deciding convention declares that it `overrides` the other. Before, the
  first convention decided without looking at the others. A user convention
  that reads a quantity differently from the FITS standard, such as a gain
  from another keyword than `GAIN`, now needs `overrides: [generic]`. Two
  zeropoints agree when they differ by at most 0.001 mag, and the note of
  the reading names the other one.
* Every flux prior that the advisor proposes states its rule and its numbers
  in the summary and in the report: the zeropoint and the bounds in image
  units, or, without a zeropoint, the sky scale, the noise-equivalent area of
  the PSF and the bounds. A prior from a declared zeropoint, and a PSF
  ceiling from a declared saturation level, have the source `declaration`;
  a prior from a zeropoint that a calibration service gave names the
  service in its detail.
* The report of a proposal has `schema_version` 3. It records the declared
  calibration with the source of each value, the user or a calibration
  service, whether the advisor was online, and, under `bridges`, every
  calibration service that serves the image with what it was asked and
  answered. A report written by an earlier version differs, so replacing it
  needs `overwrite=True` once in Python, or removing it before
  `baymorph config --advise --accept`.
* Writing a proposal again keeps an existing report that differs only in
  when a calibration service was asked, in the verbatim text of the error a
  failed request met, or in whether the advisor was online; another answer,
  or a failure for another reason, is another proposal. A refusal to replace
  a configuration or a report names the keys that differ.
* `baymorph config --advise --accept` writes neither the configuration nor
  its report when a request to a calibration service that the proposal
  needed failed, with no answer or an answer that fails its checks; run it
  again, or add `--offline` to write the proposal without it. A refusal, when
  the image allows no request or the answers hold no single value, is
  written with the priors that do without it.
* A proposal stops with an error when the image, its HDU or the declared
  calibration changes while it is built, instead of mixing two answers of a
  calibration service.
* The metadata of an observation (`ObservationMetadata`) hold only values
  that the header reading establishes or that are declared. A value whose
  unit or meaning the header leaves open, such as a saturation level in ADU
  in an image in ADU/s, is `None` there and stays in the header reading with
  its raw value; it is no longer used as if it were in the units of the
  image.

* A command that fails while it runs prints its error on one line,
  `baymorph: error: …`, and exits with status 1, also when a file is not
  FITS. Options that are invalid or do not go together still exit with
  status 2, now after the usage of their own command rather than of
  `baymorph`.
* A FITS file of 16 MiB or more is read and hashed once per session and
  shared by every stage, instead of five times and eight times for one
  saved preparation. The shared image is read-only.
* `ObservationAdvisor.header_report()` reads only the header of the image,
  never its pixels, so the report of a mosaic of several gigabytes costs the
  same as the report of a small stamp.
* A configuration that still holds the retired `data.external_mask_path`
  is refused as soon as it is read, by every command, rather than only
  when the image is prepared. Declare the curated mask under
  `data.mask_path`.
* Unknown entries in the `data` section are refused, as those of `fit`
  already were, so a misspelt key no longer passes unnoticed.
* On an image read within a working region, what covered the whole image
  covers the region: the first detection of an automatic cutout, the search
  of the PSF stars, the inspection of the observation advisor and of the
  model screening, and the check that decides whether the sky can give the
  effective gain. The interactive selection of the target still shows the
  whole image.
* The steps of a joint fit that use only the celestial WCS of an image, and
  `ObservationAdvisor.set_target_position`, read its header, not its pixels.

### Fixed

* The stars selected for an empirical PSF are reused only when the image,
  its HDU and the detection and selection settings are unchanged; they
  record them in `psf/psf_candidates_provenance.json`. Before, a run
  reused the saved stars after any change of these settings. Stars saved
  by an earlier release are reused only while the accepted PSF built from
  them still matches the current inputs.
* An accepted PSF records a declared `data.hdu`, so it is no longer reused
  for another HDU of the same file.

---

## [0.5.1] - 2026-10-08

This release lets BayMorph use the noise products of a preparation made by
an earlier release when the code that computes them is unchanged.
Configurations and products of 0.5.0 run unchanged; a checkpoint still
resumes only under the release that wrote it.

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

[Unreleased]: https://github.com/carlocannarozzophd/BayMorph/compare/v0.5.1...HEAD
[0.5.1]: https://github.com/carlocannarozzophd/BayMorph/releases/tag/v0.5.1
[0.5.0]: https://github.com/carlocannarozzophd/BayMorph/releases/tag/v0.5.0
[0.4.0]: https://github.com/carlocannarozzophd/BayMorph/releases/tag/v0.4.0
[0.3.0]: https://github.com/carlocannarozzophd/BayMorph/releases/tag/v0.3.0
