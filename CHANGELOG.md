# Changelog

All notable changes to BayMorph are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project follows Semantic Versioning in spirit.

---

## [Unreleased]

### Added

* `benchmarks/profile_likelihood_decomposition.py`, which decomposes one complete likelihood evaluation into its forward-model stages against accepted preparation and PSF products, compares convolution candidates for equivalence as well as speed, and measures the size and cost of shipping the inference problem to a worker. Executing it settled the falsification milestone required by `docs/research/PERFORMANCE_ARCHITECTURE_P0.md`: the real r-band cost is 51.5 ms per evaluation rather than the 146 ms recorded there, the recorded baseline describes the hybrid renderer which an empirical PSF never executes, `scipy.signal.oaconvolve` is 12 per cent slower than the current convolution rather than faster, a cached padded kernel FFT is bit-identical and 7.1 ms cheaper, and the bound likelihood serializes to 29.34 MB per task chunk, which accounts for the measured 21.4 per cent parallel efficiency. Results in `docs/research/LIKELIHOOD_COST_DECOMPOSITION_2026-09-20.md`.
* `preparation.masking.negative_threshold_sigma`, which excludes from the fitting mask every pixel lying more than that many noise scales below zero after background subtraction (AD-043). Source detection thresholds positive excursions and cannot exclude a negative detector artifact; in the first real production fit, seven such pixels beside a saturated star carried 39.6 percent of the total chi-square of 853 931 pixels. The criterion defaults to 5, is asymmetric by construction because no physical surface brightness is negative, reads the data and never a fitted model, and applies in every fitting-mask mode. Its threshold and excluded-pixel count are recorded in the preparation metadata.
* Explicit `background` and `psf_max_peak` declarations in `make_config_from_fits`, forwarded by `ObservationAdvisor.propose_config`. The residual-background declaration must name its model and give a typed prior for exactly the parameters that model samples; the PSF-candidate peak ceiling is a caller decision in science-image units and is never derived from a FITS saturation keyword. Proposals warn when either is left to an image-unit-agnostic default.
* `ObservationAdvisor.propose_config` records every sampled prior in the proposed configuration, including the BayMorph reference defaults of the starting model and of the residual background, and names the defaulted parameters in a warning, so a written YAML no longer depends on defaults that a later version could change.
* Observed-galaxy tutorial notebooks `02A_Observed_Assessment_and_Config` and `02B_Observed_Fit`, packaged and copied by `baymorph init`. 02A inspects one image, selects a target, and writes a reviewed configuration without preparing, building a PSF, or fitting; 02B consumes that configuration and runs preparation, PSF construction, fitting, and diagnostics without rebuilding it. Neither notebook blocks on `input()`.
* Rebuilt the simulated single-band tutorial notebook against the current public API as a linear, non-interactive Nautilus workflow that reads the packaged simulated-galaxy configuration and compares the posterior with the known simulation truth.
* Optional `truths`, `truth_color`, and `truth_linewidth` markers in `corner_plot`, applied automatically by single-observation corner diagnostics when a run has a persisted simulation truth. Truth positions are transformed into cutout coordinates, and the simulated sky is never mapped onto the residual background parameter.
* `InferenceResult.equal_weight_samples` for visualization, alongside the weighted samples that remain the statistical source of truth.
* `InferenceResult.mahalanobis_distance()` and the supporting `weighted_covariance` and `mahalanobis_distance_squared` statistics, assessing how far a reference parameter vector lies from a posterior while accounting for parameter correlations.
* Repeated-noise recovery validation campaign under `validation/repeated_noise_recovery/`, measuring posterior calibration across independent noise realizations.
* Discovered project-root-relative configurations from nested working directories and resolved their paths against the project root.
* Runner-driven all-in-one simulated-galaxy workflow and streamlined parameter-recovery tutorial.
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
* Packaging tests for runtime, development, and interactive dependencies.
* Public-source boundary checks preventing observational data and generated
  research products from entering release artifacts.

### Changed

* Joint multi-observation inference uses the residual-background abstraction of AD-031 instead of the legacy model-level `sky` parameter (AD-044). The joint `fit.background` section declares the background family, `constant` or `plane`; its parameters, `background_level` and the plane gradients, are declared under `fit.multi_observation.parameters` with independent coupling, and a configuration still naming `sky` is rejected with a migration hint. `make_multi_parameters` and `make_multi_observation_config` accept `background_model`. Existing joint checkpoints and configurations that used `sky` are incompatible.
* One production model factory, `ObservedModelFactory`, serves both single-observation and joint fits: it composes the registered morphology with the observation PSF and applies the declared residual background as an observation effect. The joint path injects each observation's PSF at call time. The `make_*_psf_sky_model` factories became `make_*_psf_model` and no longer take a `sky` argument; the `SkyModelWrapper` and the `sky` entry of every registered model contract are gone, together with the wasted full-image addition of a zero sky in every single-observation likelihood evaluation.
* `DEFAULT_CONFIG` proposes `single_sersic`, the production baseline, instead of `double_sersic`.
* The simulation configuration accepts only the nested `simulation.noise` and `simulation.psf` sections; the flat `sky`, `poisson_noise`, `read_noise_sigma` and `psf_fwhm` keys are no longer read.
* `ValidationWorkspace` writes configurations the pipeline accepts: the sampler is declared through the `init`, `run` and `execution` envelope, and the preparation section no longer carries the rejected `method` and `npixels` keys. The written configuration is now resolved in the tests.
* The likelihood alias table exists once, as `baymorph.likelihood.normalize_likelihood_type`, and the corner, residual-map and radial diagnostics share their configuration resolver, posterior accessor and title helpers through `baymorph.pipeline.diagnostics_common`.
* Validation benchmark, report and repeatability modules share the posterior-mean, evidence and likelihood-call accessors of `baymorph.validation.run`, and both validation workspaces share one case-identifier validator.
* `estimate_noise_map` fails closed when no finite unmasked pixel exists or the scale is not finite, and source detection no longer swallows a `SourceCatalog` failure into a missing catalogue.
* `pipeline_paths` no longer declares the never-written `input` directory, PSF candidate and diagnostic plots, or model-comparison summary text.
* `RealFitProducts.corner` labels the axes from the sampled parameter names, so the residual-background parameters are labelled correctly and a plane background no longer raises.
* Real-object configurations under `configs/` are ignored by Git; the object-neutrality guard also recognises identifiers carrying a band or version suffix and rejects local absolute paths in tracked text; the observed-galaxy tutorials use the neutral `galaxy_r` placeholder for the FITS image, the analysis name and the written configuration.
* PSF convolution precomputes the padded kernel transform, the transform shape and the crop indices once per PSF and image shape, instead of transforming a constant 25 x 25 kernel on every likelihood evaluation, and `PSFEffect.apply` no longer revalidates a kernel its constructor already validated. The cached path performs the same operations in the same order as `scipy.signal.fftconvolve` and is bit-identical to it, not merely equal within a tolerance. On the r-band production problem this takes convolution from 17.6 ms to 11.1 ms, a complete likelihood evaluation from 51.5 ms to 46.4 ms, and halves the convolution's peak allocation from 35.7 MB to 17.8 MB. `scipy.signal.oaconvolve` was measured on the same problem and rejected: it is 12 per cent slower.
* The Nautilus backend hands the sampler a module-level likelihood function and installs the inference problem in each worker through the pool initializer, as the emcee backend already did. Nautilus receives an `mp.Pool` object rather than an integer, so its own worker-caching path is never taken and `Pool.map` re-serializes the mapped callable once per task chunk; with a bound method that dragged the entire `ImageInferenceProblem`, measured at 29.34 MB per chunk, or roughly 9.8 MB per likelihood evaluation, for the whole run. What now travels per chunk is 64 bytes. Serial execution is unchanged and still uses the bound method. Dynesty and pocoMC have the same defect and are not addressed here.
* Image-preparation products created before AD-043 no longer match their inputs, because `preparation.masking` is part of the preparation configuration identity. They are rejected fail-closed with a message directing the user to a new output namespace. This is intended: their likelihood support admits pixels the current preparation excludes, so evidence computed on them is not comparable with evidence computed now. No existing artifact is mutated.
* The error rejecting fixed noise-estimation entries no longer describes the production noise estimator as MAD-based, which AD-041 superseded when the scale moved to residuals about a plane fitted to the sky region.
* `BayMorphRunner.prepare_image`, `select_psf_candidates` and `build_psf` reuse an accepted product when one exists for the configured output namespace, instead of failing on its immutability, and report the reuse (AD-042). Loading remains fail-closed: products whose immutable inputs or provenance no longer match are rejected. `outputs.overwrite` still selects rebuilding, and the PSF candidate catalogue is now protected by that policy like the PSF itself.
* `ObservationConfigProposal.write` refuses to replace an existing configuration unless `overwrite=True` is passed, and the Advisor summary reports `BUNIT` and the saturation keyword as header metadata that is not unit-checked while labelling its background and noise scales as whole-image diagnostics.
* Aligned single-image colorbars with their image panels and standardized pixel-axis and colorbar styling.
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
* Adopted normalized Gaussian likelihoods as neutral tutorial defaults while
  requiring explicit validation for correlated-noise models.

### Removed

* The `sky` model-level parameter, `SkyModelWrapper`, the prior-boundary alias mapping `background_level` to `sky`, and the `sky` prior requirement of the prior-predictive flux calibration, which demanded a prior that the configuration contract forbids and therefore always failed.
* The `data.external_mask_path` and `data.cutout_size` configuration shims, the `profile_noise` argument of the radial-profile plot, and the readable legacy (`format_version`-less) inference archives, which now fail closed with a message asking for a rerun.
* Experimental native nested-sampler runtime and its production integration.
* Superseded geometry-specific sampler infrastructure.
* Legacy multiband compatibility modules and obsolete configuration paths.
* Obsolete validation backends, notebooks, audit artifacts, and broken
  benchmarks.
* Target-specific public configuration and tracked generated benchmark tables,
  research summaries, and figures.
* All tutorial notebooks, pending a rebuild against the current public API.

### Fixed

* `ObservationAdvisor.calibrate_flux_prior` and the single-Sérsic prior-predictive diagnostics no longer demand a `sky` prior, so the calibration runs on every valid configuration.
* The multi-observation validation workspace, SBC reference contract and preparation-free recovery campaign declare the residual background as `background_level`, matching the production contract.
* The prepared noise scale is now measured from residuals about a plane fitted
  to the sky region instead of about a constant, so an unmodelled sky gradient
  is no longer counted as noise: a ramp of plus or minus one count on a scale
  of two inflated the estimate by 4.1 percent, and three counts by 33.8
  percent. Preparation still subtracts a constant. An unmodelled gradient now
  surfaces as an elevated reduced chi-squared rather than being absorbed; the
  remedy is `fit.background.model: plane`. See AD-041.
* Preparation reports the fitted sky gradient, its standard error and its
  significance per axis, in the preparation metadata and the runtime report.
  It is a diagnostic and never selects a background model.
* `pipeline_paths` exposes `fit_summary` and `fit_summary_text` in place of a
  `fit_results` key that named a file nothing ever wrote; the fit summaries are
  now written through those keys instead of being composed inline.
* The prepared noise field of the canonical single-observation workflow is now
  a truncation-corrected Gaussian scale instead of a scaled median absolute
  deviation. The previous estimator was not materially biased but was
  inefficient, carrying about 1.1 percent run-to-run scatter against 0.7
  percent now; that scatter enters the likelihood normalization and so sets
  every credible-interval width and displaces absolute evidence. Re-deriving
  the canonical example from its persisted sky region moves the scale from
  1.9628 to 2.0107 against a true 2.0, and the implied reduced chi-squared
  from 1.0470 to 0.9977. See AD-040.
* `estimate_noise_map`, which serves the observation advisor and
  empirical-PSF candidate selection rather than the likelihood, now uses a
  scaled median absolute deviation of the clipped sample instead of its plain
  standard deviation, improving its bias from −1.54 to −0.26 percent.
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
* Enlarged the simulated-galaxy example field to 121 x 121 pixels so that it
  contains the sky plateau plus the preparation sky margin; the shipped
  81 x 81 example failed during image preparation.
* Residual-map diagnostics no longer load empirical-PSF provenance for fits
  that use an analytic Gaussian PSF, which made `run(diagnostics=True)` fail
  after the fit; empirical-PSF validation is unchanged.

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
