# Python API

Everything an analysis needs is imported from the top-level `baymorph`
package, and the figures from `baymorph.plotting`. The other modules, such
as `baymorph.pipeline`, `baymorph.fitting` or `baymorph.rendering`, are the
internals of the package and may change between versions.

## Running an analysis

**`BayMorphRunner(config, *, verbose=True)`** runs an analysis from a YAML
path or a configuration dictionary. See [The Runner](../guide/RUNNER.md).

| Member | Description |
| --- | --- |
| `run(*, diagnostics=False)` | Run every stage; returns the fit products. |
| `simulate()` | Make the synthetic image of a `simulation` section. |
| `prepare_image()` | Cut out, mask, subtract the sky, measure the noise. |
| `select_psf_candidates()`, `build_psf()` | Select the PSF stars and build the kernel. |
| `fit()` | Sample the posterior; a dictionary by model name for several models. |
| `diagnostics()`, `corner()`, `residual_maps()`, `radial_profile()` | Write the diagnostic figures. |
| `image`, `psf_candidates`, `psf`, `results`, `simulation` | The products of the stages already run. |
| `run_dir` | The run directory, `outputs.root/object.name`. |

## Writing configurations

**`make_config_from_fits(fits_path, *, target_position, ...)`** returns a
configuration for one FITS image, and writes it when `output_path` is
given. Its keyword arguments are `hdu`, `calibration`, `region`, `output_path`,
`object_name`, `model_name`, `model_priors`, `likelihood`, `background`,
`sampler`, `outputs_root`, `overwrite`, `cutout_mode`, `fixed_size`,
`psf_type`, `gaussian_fwhm_pixels`, `psf_max_peak` and `preparation`. See
[Writing a Configuration](../guide/CONFIGURING.md).

**`ObservationAdvisor(fits_path, *, target_position=None, hdu=None, calibration=None, online=True, region=None)`**
inspects one image and proposes a configuration for review. `target_position`
is `(x, y)` or `{"x": ..., "y": ...}` in pixels of the whole image, `"center"`,
or `{"ra": ..., "dec": ...}` on the sky (ICRS), which the advisor converts to
pixels through the celestial WCS of the header; `make_config_from_fits`
takes the same forms and writes pixels. `calibration`
declares calibration quantities, `online=False` keeps it from asking a
calibration service for a quantity that the header lacks, and `region` sets
the working region of a large image, as `data.region`: the inspection reads
only that region, about the target or, before a target is set, about the
centre of the image.

| Member | Description |
| --- | --- |
| `ObservationAdvisor.from_project(fits_path)` | Create an advisor, with paths relative to the project; it takes the same keyword arguments. |
| `header_report()` | The quantities read from the header, with their values, units, statuses and sources, as a table to print, and what each calibration service that serves the image was asked and answered. |
| `select_target()`, `set_target_position(x, y)` | Choose the target, by clicking or explicitly. |
| `inspect()`, `summary()` | Measure and print header metadata and image diagnostics. |
| `prior_guidance(model_name)` | The prior family of every parameter of any of the ten models and the rule that proposes it, as `PriorGuidance` objects. |
| `measure_fixed_inputs(...)` | Prepare the image and build its PSF, and return the sky scale, the cutout and the FWHM of the PSF as `FixedInputs`. |
| `propose_config(...)` | Propose every prior of one model (`model_name`) or of several (`models`) from the header, physics and the fixed inputs, and build an `ObservationConfigProposal` in memory. |

`ObservationConfigProposal.summary()` lists the proposal, the source of
every prior and its warnings, `config` holds it, `models` names its
models, `provenance` records where each value comes from, and
`write(accept=True)` writes the YAML with the report `report_path` beside
it. Writing again keeps a report that differs only in when a calibration
service was asked, in the verbatim text of the error a failed request met,
or in whether the advisor was online; an existing different file is
replaced only with `overwrite=True`, and the refusal names the keys that
differ.

**`make_multi_parameters(model, observations, *, target_centers, prior_bounds, shared, ...)`**
and **`make_multi_observation_config(observations, *, target_positions, parameters, output_dir, ...)`**
build a joint analysis. See [Several Observations](../guide/MULTI_OBSERVATION.md).

## Results

**`RealFitProducts`** is what `fit()` returns for one model.

| Member | Description |
| --- | --- |
| `posterior` | The `InferenceResult`. |
| `best_parameters` | The parameters of the sample of highest likelihood. |
| `best_model_image`, `residual`, `norm_residual`, `noise` | The model, the residuals and the sky noise, on the cutout. |
| `chi2`, `ndof`, `reduced_chi2` | The goodness of fit. |
| `summary()` | Model, statistics, evidence, best parameters and noise model. |
| `corner(**kwargs)` | Draw the corner plot. |
| `radial_profile()` | Draw the radial profile, once the diagnostics have measured it. |

**`InferenceResult`** holds the weighted posterior and the evidence, whatever
the sampler.

| Member | Description |
| --- | --- |
| `parameter_names` | The parameters, in sampling order. |
| `samples`, `weights` | The weighted samples and their normalized weights. |
| `equal_weight_samples` | Equal-weight samples derived from them, for plots. |
| `summary(name)` | Weighted median and 16th and 84th percentiles of one parameter. |
| `median_parameters`, `best_parameters` | Posterior medians, and the sample of highest likelihood. |
| `effective_sample_size` | The effective number of independent samples. |
| `logz`, `logz_error`, `evidence_method` | The natural log of the evidence, its uncertainty and how it was computed. |
| `mahalanobis_distance(reference)` | The distance of a reference point, such as a simulation truth, from the posterior, accounting for correlations. |
| `save(path)`, `InferenceResult.load(path)` | Write and read a posterior file, such as `posterior.npz`. |

`PosteriorSamples` and `EvidenceEstimate` are the two parts of an
`InferenceResult`, available as its `posterior` and `evidence` attributes.

## Simulations

| Name | Description |
| --- | --- |
| `SyntheticSersic(flux, re, n, q, theta, x0, y0)` | A Sérsic galaxy to simulate; `theta` in radians. |
| `make_observation(galaxy, shape, sky, psf_fwhm, poisson_noise, read_noise_sigma, seed)` | A synthetic image, returned as a `SyntheticObservation`. |
| `SyntheticObservation` | The image, its noise-free stages and the truth; `to_fits(path)` writes it. |
| `GaussianPSF(fwhm)`, `EmpiricalPSF(kernel, ...)` | PSF objects. |

See [Simulated Images](../guide/SIMULATIONS.md).

## Plotting

From `baymorph.plotting`; see [Plotting](../guide/PLOTTING.md).

| Function | Draws |
| --- | --- |
| `plot_log_image(image, title="", ...)` | One image on a logarithmic scale. |
| `plot_log_data_model_residual(data, model, *, norm_residual=None, ...)` | Data, model and residuals. |
| `corner_plot(samples, parameter_labels=None, truths=None, ...)` | A corner plot. |
| `plot_radial_profile_diagnostics(radius, profile_data, profile_model, ...)` | A radial profile with its residuals. |
| `make_plot(nrows=1, ncols=1, ...)` | A figure in the BayMorph style. |
| `set_plot_style()`, `style_axes(ax)` | Apply the BayMorph style. |

## Advanced

The model and data classes `SersicModel`, `DoubleSersicModel`,
`CompositeModel`, `ConvolvedModel`, `Observation`, `ObservationMetadata` and
`ImageData` let you render models and handle images directly, outside a
configured analysis.

The model-screening interface, `SingleObservationModelAdvisor` with
`SingleObservationScreeningResult`, `ModelScreeningReport`,
`ModelScreeningScore`, `ModelScreeningClassifierArtifact` and
`load_model_screening_classifier`, ranks models from image features with a
trained classifier. No validated classifier is distributed with BayMorph,
and a screening score is never a Bayesian evidence: the model to fit stays
your decision.

## Stability

The names on this page are the public interface: they change only with a
documented migration. A change that corrects a scientifically wrong
behaviour takes priority over compatibility.
