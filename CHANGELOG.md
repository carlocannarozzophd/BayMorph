# Changelog

All notable changes to BayMorph are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project follows Semantic Versioning in spirit.

---

## [Unreleased]

### Added

* AD-048, six further morphology families as available single-observation
  models, and the survey behind it
  ([note](docs/research/MORPHOLOGY_FAMILIES_2026-10-02.md)): a catalogue of
  families beyond the Sérsic profile, spiral arms included, and the reasons
  why the production target needs none of them.
* A Models section on the documentation site (`docs/models/`): the morphological
  components, the four morphology models, the PSFs and the residual background,
  each with its definition, parameters, priors, degeneracies, validation status
  and figures in a light and a dark version from
  `scripts/make_model_figures.py`: most are computed with BayMorph, one
  evaluates the Sérsic equations with SciPy and two are diagrams of
  conventions. Tests keep the pages, models and figures in step.
* A user guide on the documentation site (`docs/guide/`): installation, quick
  start, command line, configuration reference, preparation and products, and
  tutorials. `tests/test_documentation.py` keeps it in step with the default
  configuration, the models and the command line.
* The proposed AD-047, source variance and sky correlation in the noise model
  ([note](docs/research/NOISE_MODEL_PROPOSAL_2026-10-02.md)); it enters
  `DECISIONS.md` only once approved.
* A simulation bench of noise models (`benchmarks/noise_model_bench.py`): the
  production model reports intervals up to 3.0 times too narrow, a model with
  source variance and sky correlation within 0.90-1.16
  ([note](docs/research/NOISE_MODEL_BENCH_2026-10-01.md)).
* A feasibility study of computer-vision source identification against the
  production detection (`benchmarks/cv_source_identification.py`), closed with
  a no-go as a replacement
  ([note](docs/research/CV_SOURCE_IDENTIFICATION_2026-10-01.md)).
* An audit of the noise and background chain with a sandwich measurement of
  interval calibration (`benchmarks/noise_interval_calibration.py`): the
  production noise model reports intervals of n and re four to five times too
  narrow ([note](docs/research/NOISE_BACKGROUND_AUDIT_2026-10-01.md)).
* The proposed AD-046, effective-PSF rendering with extended empirical kernels
  ([note](docs/research/PSF_RENDERING_PROPOSAL_2026-10-01.md)), since
  accepted.
* Empirical-PSF builders with extended wings measured on the test bench
  (`benchmarks/psf_wing_candidates.py`); W1, with the wings of the stars
  measured out to 64 px, keeps the bias of n and re within 1 percent
  ([note](docs/research/PSF_EXTENDED_WINGS_2026-10-01.md)).
* A measurement of how PSF and rendering errors move the fitted Sérsic
  parameters (`benchmarks/psf_parameter_sensitivity.py`): sampling at pixel
  centres moves n and re by up to 14 and 22 percent
  ([note](docs/research/PSF_PARAMETER_SENSITIVITY_2026-10-01.md)).
* A comparison of rendering conventions under an empirical PSF against a
  brute-force reference (`benchmarks/rendering_convention.py`): sampling at
  pixel centres fails both criteria, a smooth-window local oversampling meets
  both ([note](docs/research/RENDERING_CONVENTION_2026-10-01.md)).
* Candidate empirical-PSF builders measured on the test bench
  (`benchmarks/psf_builder_candidates.py`): a phase regression of the stars,
  with no interpolation, meets every criterion
  ([note](docs/research/PSF_BUILDER_CANDIDATES_2026-09-30.md)).
* A test bench for the empirical-PSF builder (`benchmarks/psf_test_bench.py`):
  the production kernel failed its own centring check in 36 of 40 fields
  ([note](docs/research/PSF_TEST_BENCH_2026-09-30.md)).
* An adaptive model-informed evidence estimator compared with Nautilus,
  Dynesty and pocoMC on six analytic targets: Nautilus was correct in 60 of 60
  runs, the estimator in 47
  ([note](docs/research/STRUCTURE_INFORMED_EVIDENCE_ADAPTIVE_2026-09-30.md)).
* Likelihood views extending the model-informed evidence estimator to the
  correlated Gaussian, Cash and Poisson-Gaussian likelihoods
  (`benchmarks/structure_informed_likelihood_views.py`)
  ([note](docs/research/STRUCTURE_INFORMED_EVIDENCE_LIKELIHOODS_2026-09-29.md)).
* A comparison of the model-informed evidence estimator with Nautilus on the
  simulated galaxy of the first tutorial
  (`validation/structure_informed_evidence/`), a smoke test on one galaxy.
* A model-informed evidence estimator that reproduces the Nautilus evidence of
  the r-band fit with 15 times fewer evaluations and flags its own failures;
  AD-045 remains a proposal
  ([note](docs/research/STRUCTURE_INFORMED_EVIDENCE_2026-09-29.md)).
* A decomposition of the cost of one likelihood evaluation
  (`benchmarks/profile_likelihood_decomposition.py`): 51.5 ms on the real
  r-band problem, not the recorded 146 ms
  ([note](docs/research/LIKELIHOOD_COST_DECOMPOSITION_2026-09-20.md)).
* `preparation.masking.negative_threshold_sigma`, default 5, which excludes
  impossibly negative pixels from the likelihood support (AD-043).
* Explicit `background` and `psf_max_peak` declarations in
  `make_config_from_fits` and `ObservationAdvisor.propose_config`, which warn
  when either is left to a default not scaled to the image.
* `ObservationAdvisor.propose_config` writes every sampled prior, defaults
  included, and names the defaulted ones in a warning.
* Observed-galaxy tutorial notebooks `02A_Observed_Assessment_and_Config` and
  `02B_Observed_Fit`, packaged and copied by `baymorph init`.
* The simulated single-band tutorial notebook, rebuilt against the current
  public API.
* Truth markers in `corner_plot`, drawn automatically for simulated
  single-observation runs.
* `InferenceResult.equal_weight_samples` for visualization, beside the
  weighted samples that remain the statistical source of truth.
* `InferenceResult.mahalanobis_distance()`, with `weighted_covariance` and
  `mahalanobis_distance_squared`, to measure how far a reference vector lies
  from a posterior.
* A repeated-noise recovery campaign under
  `validation/repeated_noise_recovery/`, measuring posterior calibration
  across independent noise realizations.
* Discovery of project-root-relative configurations from nested working
  directories.
* A runner-driven all-in-one simulated-galaxy workflow and parameter-recovery
  tutorial.
* A prior-volume coverage review for observationally calibrated flux priors,
  and post-fit diagnostics of posterior contact with prior boundaries.
* Canonical multi-observation configuration, preparation, joint inference,
  persistence and diagnostics.
* Explicit parameter coupling across observations: exact sharing,
  observation-specific and fixed parameters, hierarchical and
  wavelength-dependent coupling.
* Native-grid and WCS-aware modelling of heterogeneous observations.
* The Nautilus production backend beside Dynesty and pocoMC.
* Within-observation correlated-noise calibration and correlated Gaussian
  image likelihoods.
* Likelihood normalization status and per-observation likelihood provenance.
* Hybrid Sérsic rendering with validated adaptive numerical refinement.
* The observation advisor, guided configuration and evidence-aware flux-prior
  calibration.
* Experimental preparation-free model screening with simulation, interpretable
  features, a classifier and mandatory user model selection.
* Backend-neutral validation runners, reference problems, repeatability
  analyses and simulation-based calibration infrastructure.
* Packaging tests for runtime, development and interactive dependencies.
* Public-source boundary checks that keep observational data and generated
  research products out of release artifacts.

### Changed

* `README.md`, `CONTRIBUTING.md`, `ROADMAP.md` and the ADR policy are revised,
  `DECISIONS.md` gains an index, and the Unreleased entries of this changelog
  are shortened.
* The documentation site is reorganized: a Getting Started section, research
  notes grouped by workstream, a complete architecture map, and retitled
  validation documents. `PAPER_BLUEPRINT.md` moved to
  `docs/archive/research_history/` with a status note.
* The architecture documents follow AD-046: `docs/architecture/RENDERING.md`
  §13.1 defines the effective-PSF rendering in place of the rule that an
  empirical kernel requires sampling at pixel centres.
* The rendering of AD-046 is evaluated within 2e-10 of its previous evaluation
  in 68-72 ms instead of 158 ms per complete likelihood evaluation of a
  production-size problem; checkpoints made with the previous evaluation are
  rejected ([note](docs/research/EFFECTIVE_PSF_COST_2026-10-01.md)).
* The production empirical-PSF builder is the builder of AD-046, with four new
  keys of `psf.empirical.construction` and PSF products of schema 2 holding
  the sub-pixel kernels; products of the previous schema are rejected for
  fits.
* `EmpiricalPSF` accepts sub-pixel kernels and `ConvolvedModel` renders with
  them as AD-046 defines; the rendering convention enters every checkpoint
  signature.
* Three duplications are merged without changing any digest or signature:
  sampler checkpoint signing, the backend envelope, and canonical JSON
  hashing.
* The current priority is stated once, in `ROADMAP.md`, and the other
  documents refer to it.
* Joint inference uses the residual-background abstraction instead of the
  model-level `sky` parameter (AD-044); configurations naming `sky` are
  rejected with a migration hint.
* One production model factory, `ObservedModelFactory`, builds the observed
  model of single and joint fits.
* `DEFAULT_CONFIG` proposes `single_sersic`, the production baseline, instead
  of `double_sersic`.
* The simulation configuration accepts only the nested `simulation.noise` and
  `simulation.psf` sections.
* `ValidationWorkspace` writes configurations the pipeline accepts.
* The likelihood alias table and the diagnostic helpers exist once.
* The validation modules share their posterior, evidence and likelihood-call
  accessors.
* `estimate_noise_map` fails closed when no finite unmasked pixel exists, and
  source detection no longer swallows a `SourceCatalog` failure.
* `pipeline_paths` no longer declares directories and files that are never
  written.
* `RealFitProducts.corner` labels its axes from the sampled parameter names.
* Real-object configurations under `configs/` are ignored by Git, and the
  object-neutrality and local-path guards cover more cases.
* PSF convolution caches the padded kernel transform, bit-identical to
  `scipy.signal.fftconvolve`: a complete r-band evaluation takes 46.4 ms
  instead of 51.5 ms.
* The Nautilus backend installs the inference problem in each worker once, so
  64 bytes instead of 29.34 MB travel per task chunk; Dynesty and pocoMC keep
  the defect.
* Image-preparation products made before AD-043 are rejected, because
  `preparation.masking` enters the preparation identity.
* The error rejecting fixed noise-estimation entries no longer calls the
  production noise estimator MAD-based.
* `prepare_image`, `select_psf_candidates` and `build_psf` reuse an accepted
  product of the same output namespace and report the reuse (AD-042).
* `ObservationConfigProposal.write` refuses to replace an existing
  configuration unless `overwrite=True` is passed.
* Single-image colorbars align with their image panels.
* Legacy multiband terminology and runtime objects are replaced by the general
  multi-observation architecture.
* Inference backends consume only backend-neutral `InferenceProblem`
  instances.
* Posterior samples, evidence, diagnostics, timing and provenance live in
  backend-neutral result objects.
* Inference footprints are separate from preparation footprints.
* Typed prior configuration and parameter routing are standardized.
* Plotting no longer needs an external LaTeX installation.
* Model-screening runtime code is separate from its validation and training
  infrastructure.
* The documentation is consolidated around the production contracts.
* The roadmap is refocused on real multi-observation production fits and paper
  closure.
* Normalized Gaussian likelihoods are the neutral tutorial default, and
  correlated-noise models require explicit validation.

### Removed

* Code used only by tests, found by the deep audit of 2026-09-30, with its
  tests: five modules and sixteen functions, none of them in the top-level
  public API ([audit](docs/archive/audits/BayMorph_DEEP_AUDIT_2026-09-30.md)).
* The posterior-only emcee backend, never released. Every fit, in both
  workflows, now requires a completely normalized likelihood.
* Dead or superseded code found by the audit of 2026-09-29.
* The model-level `sky` parameter and `SkyModelWrapper`.
* The `data.external_mask_path` and `data.cutout_size` shims, and legacy
  inference archives without a format version.
* The experimental native nested-sampler runtime and its production
  integration.
* Superseded geometry-specific sampler infrastructure.
* Legacy multiband compatibility modules and obsolete configuration paths.
* Obsolete validation backends, notebooks, audit artifacts and broken
  benchmarks.
* Target-specific public configuration and tracked generated tables, summaries
  and figures.
* All tutorial notebooks, pending a rebuild against the current public API.

### Fixed

* Source detection no longer falls back silently to a background that ignores
  the detection mask: preparation detects without a background, and the
  observation advisor raises instead.
* Documents that declared capabilities the code does not provide, found by the
  deep audit of 2026-09-30, with status notes under AD-028, AD-032, AD-040 and
  AD-042.
* AD-041 names the project's scientific invariants instead of a file that is
  not distributed.
* `ObservationAdvisor.calibrate_flux_prior` no longer demands a `sky` prior.
* The multi-observation validation workspace and campaigns declare the
  residual background as `background_level`.
* The prepared noise scale is measured about a plane fitted to the sky region,
  so an unmodelled gradient is not counted as noise (AD-041).
* Preparation reports the fitted sky gradient and its significance as a
  diagnostic that never selects a background model.
* `pipeline_paths` exposes `fit_summary` and `fit_summary_text` in place of a
  key naming a file that was never written.
* The prepared noise scale is a truncation-corrected Gaussian scale instead of
  a scaled median absolute deviation, with 0.7 instead of 1.1 percent
  run-to-run scatter (AD-040).
* `estimate_noise_map`, then used by the observation advisor and the
  empirical-PSF candidate selection, uses a scaled median absolute deviation:
  bias -0.26 instead of -1.54 percent.
* Per-observation likelihood overrides take precedence over joint defaults.
* The correlated-noise calibration is built before joint inference for every
  observation that needs it.
* Multi-observation fit statistics use each observation's likelihood and
  noise.
* Joint products persist the per-observation likelihood metadata and
  normalization status.
* Mahalanobis statistics are consistent with the likelihood of correlated
  observations.
* All runtime dependencies are declared, and the interactive and development
  ones are optional.
* PSF selection, mask handling, background estimation, radial diagnostics,
  output persistence and sampler provenance.
* Inconsistencies of the public notebooks, API, terminology and release
  documentation.
* The simulated-galaxy example field is 121 x 121 px, so that it contains the
  sky plateau and the preparation margin.
* Residual-map diagnostics no longer load empirical-PSF provenance for fits
  with an analytic Gaussian PSF.

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
