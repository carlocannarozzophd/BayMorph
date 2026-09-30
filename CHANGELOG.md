# Changelog

All notable changes to BayMorph are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project follows Semantic Versioning in spirit.

---

## [Unreleased]

### Added

* An adaptive version of the model-informed evidence estimator, in `benchmarks/structure_informed_adaptive.py`, and a campaign comparing both estimators with Nautilus, Dynesty and pocoMC on six analytic targets with difficult geometry, in `benchmarks/structure_informed_targets.py`. The adaptive estimator fits a mixture of Gaussians to its pooled weighted draws, freezes the proposal that is most efficient on all the draws made so far, and takes the evidence and the posterior from a fresh draw of that proposal only, which keeps the correctness of plain importance sampling. Every target reference is computed without the compared methods and checked by a second independent computation. On ten seeds per pair, with design and criteria fixed before the runs, Nautilus was correct in 60 of 60 runs, pocoMC in 59, the adaptive estimator in 47 and the first-spike estimator in 27. The adaptive estimator handles curved two-dimensional, ring-shaped, multimodal and skewed posteriors with the least evidence work on three targets, but fails a curved posterior in ten dimensions, and three of its sixty runs were wrong without being flagged. The first-spike estimator remains the most efficient where the posterior is close to Gaussian, and missed one of four separated modes in half its runs without any diagnostic seeing it. The Laplace geometry of the estimator now takes one-sided differences at a prior bound, where it previously took central differences with steps of a few units in the last place. Nothing in the package changes and AD-045 remains a proposal. Results and limits are in `docs/research/STRUCTURE_INFORMED_EVIDENCE_ADAPTIVE_2026-09-30.md`.
* Likelihood views for the model-informed evidence estimator, in `benchmarks/structure_informed_likelihood_views.py`, which extend it from the diagonal Gaussian likelihood to the Vecchia correlated Gaussian likelihood, exactly, and to the Cash and Poisson-Gaussian likelihoods through a variance function and iteratively reweighted least squares. The importance weights now evaluate the production likelihood of the problem, so that a view shapes the proposal only and an approximate view costs efficiency, never correctness. A likelihood without a view is refused: there is no black-box fallback. On synthetic problems of 31 by 31 pixels, against Nautilus with 1000 live points, all 29 high-signal problems of the four families meet the four acceptance criteria of the first spike with 9.5 to 13 times fewer evaluations, and all 7 low-signal controls fail and are flagged by the estimator itself. The design and the criteria were fixed before the runs. The campaign is synthetic, uses small images and a pixel-integrated forward model, and has four realizations with read noise. Nothing in the package changes and AD-045 remains a proposal. Results, limits and three findings that concern production are in `docs/research/STRUCTURE_INFORMED_EVIDENCE_LIKELIHOODS_2026-09-29.md`.
* Comparison campaign under `validation/structure_informed_evidence/`, which fits the simulated galaxy of the first tutorial with the experimental model-informed evidence estimator and with Nautilus on the same prepared problem, and compares evidence, cost, weighted posteriors and model images. The estimator runs first and reads only the prepared image products; a separate step then confirms that both fits sampled the same problem, by evaluating the likelihood of the best Nautilus sample on the problem the estimator used. `HybridResult` records the log-likelihood of every draw, so that the estimator output can be expressed in the production `InferenceResult` container and reuse the production summaries and plots. A regression test verifies that the likelihood the estimator weighs is the production likelihood, with and without a mask. On the reference run the evidences differ by 0.74 sigma, with 9061 likelihood evaluations against 81 500. This is a smoke test on one galaxy; the estimator remains research code outside the package and the notebook is committed unexecuted.
* `benchmarks/structure_informed_evidence_probe.py`, a falsification spike for a model-informed evidence estimator that exploits three properties of BayMorph morphology problems unavailable to a black-box sampler: the Gaussian likelihood gives the Gauss-Newton Hessian from first derivatives of the model image, the posterior mode and that Hessian define a defensive Student-t proposal whose importance weights preserve the declared prior and the evidence exactly, and fluxes and background coefficients are linear and can be integrated analytically. On the completed r-band production fit the estimator reproduces the Nautilus evidence within 1.1 sigma and every posterior mean and width, with 22 243 likelihood evaluations against 333 500. On twenty synthetic high-signal realizations it agrees with Nautilus in every case with 12.7 times fewer evaluations. It fails, and reports its own failure through the Pareto shape and the effective sample size, at low signal, on the two-component model and when a point-sampled cusp lies on a pixel centre. Nothing in the package changes. Results and the pre-registered acceptance criteria are in `docs/research/STRUCTURE_INFORMED_EVIDENCE_2026-09-29.md`.
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

* The current priority is stated once, in `ROADMAP.md`, which records the order set after the deep audit of 2026-09-30: revise the observation model, then the first validated real single-observation fit, then the multi-observation fit, then the paper and its release. `README.md`, `docs/index.md` and `docs/architecture/BAYMORPH_CORE_CONTRACT.md` refer to it instead of stating three different priorities.
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

* The posterior-only emcee backend, archived after the deep audit of 2026-09-30 (`docs/archive/audits/BayMorph_DEEP_AUDIT_2026-09-30.md`). It was added after 0.3.0 and never released. It produced no Bayesian evidence, no validation campaign or benchmark exercised it, and its example configuration ended without reaching its own convergence criterion at the shipped settings. Removed with it: `EmceeFitter`, `EmceeBackend` and its registration, the embedded HDF5 checkpoint and its signature helper `resolve_embedded_signed_checkpoint`, the emcee sampler defaults, the emcee branches of the single-observation and joint fits, of the configuration proposal summary and of `InferenceResult.effective_sample_size`, `configs/simulated_galaxy_emcee.yaml`, the `emcee` dependency and their tests. `h5py` stays, because Nautilus checkpoints need it. A configuration naming `emcee` is now rejected as an unknown sampler. The code remains in the Git history. Consequence: emcee was the only backend exempt from the requirement of a completely normalized likelihood, so every fit, in both workflows, now requires one and a configuration without it is rejected before any output is written. This excludes the exact correlated Gaussian likelihood, which is not normalized, the Gaussian likelihood with `include_normalization: false`, and the Cash likelihood unless `include_data_term: true`. Regression tests check that none of the three remaining backends is exempt, in either workflow. Nautilus remains the default backend, with Dynesty and pocoMC as alternatives, as AD-039 states.
* Dead or superseded code identified by the 2026-09-29 audit: the `SkyEllipseParameterTransform` module, the mock fit report (`plot_mock_fit_report` and `RealFitProducts.report`), the one-dimensional grid search, the interactive `ObservationAdvisor.configure` flow with its prompt and audit printers, the separate lower- and upper-bound flux calibrators superseded by `calibrate_single_sersic_flux_prior`, the legacy inference-archive loader, the unused `_run_sampler` helper, `ParameterSet.complete_values`, `MultiObservationParameterSpec.observation_specific`, `wavelength_correlation_matrix`, `detect_sources_mask`, and the scalar-sky fallback of the radial diagnostics, which now require an observed model exposing `render_background()`.
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

* Documents that declared capabilities the code does not provide, found by the deep audit of 2026-09-30 (`docs/archive/audits/BayMorph_DEEP_AUDIT_2026-09-30.md`). `README.md` and `ROADMAP.md` now state that the within-observation correlated-noise likelihood is available in the multi-observation workflow only: the single-observation workflow does not pass the nearest-neighbour factorization to its likelihood, and the exact correlated likelihood is not normalized. `docs/API.md` no longer lists `fit()` among the stages that reuse a completed product: it raises `FileExistsError` on a completed fit, and no public stage reloads one. The sampler examples of `README.md`, `docs/API.md` and `docs/architecture/INFERENCE.md` are nested under `fit`, where the configuration reads them; copied as they were written, at the top level, they were ignored and the default backend was used. Status notes, which leave the accepted text unchanged, record under AD-028 that the Sérsic flux is normalized analytically over the full plane, under AD-032 that AD-039 supersedes it, under AD-040 that joint inference takes its noise from the prepared noise field of each observation, and under AD-042 that a completed fit is not reused. `docs/architecture/BAYMORPH_CORE_CONTRACT.md` closes a code fence that swallowed the rest of the page and marks its g-band cost baseline as historical, next to the measured r-band cost; `docs/architecture/PRODUCTION_WORKFLOW.md` no longer describes the existing joint problem class as future work; `docs/research/PERFORMANCE_ARCHITECTURE_P0.md` carries a status note pointing to the cost decomposition that falsified its baseline. No code changes.
* AD-041 names the rules it relies on as the project's scientific invariants instead of a local file that is not distributed. The decision and its rationale are unchanged.
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
