# Configuration Reference

A BayMorph analysis is a YAML mapping. BayMorph merges it with its defaults,
validates it, and writes the resolved configuration into the run directory as
`config_resolved.yaml`. This page lists every section and key of a
single-observation configuration. The joint configuration of several
observations is described in
[Multi-Observation Workflow](../architecture/MULTI_OBSERVATION_PUBLIC_WORKFLOW.md).

Three keys are required: `object.name`, `data.image_path` and `outputs.root`.
Unknown entries in the `fit` section are rejected. An unknown top-level
section is not read: a `sampler` block written at the top level, for
instance, is ignored, and the default backend is used.

## What a change invalidates

| Keys | Enter | Effect of a change |
| --- | --- | --- |
| `object.name`, `data.*`, `preparation.*` | the identity of the prepared products | the products of the namespace no longer match and are rejected; use a new namespace or `outputs.overwrite` |
| `psf.empirical.*` | the identity of the empirical PSF | as above, for the PSF product |
| `fit.*` | the signature of the fit checkpoint | a checkpoint made with other settings is not resumed |
| `diagnostics.*` | nothing scientific | only how completed products are displayed |

## `object`

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `object.name` | string | required | Name of the analysis. Products are written to `outputs.root/object.name/`. |

## `data`

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `data.image_path` | path | required | FITS image to analyse. |
| `data.target_position` | `{x, y}` or `center` | `null` | Target-identification seed in zero-based full-image pixel coordinates, or the detector centre. Required to prepare and fit; it is also the reference of priors declared with `relative_to`. |
| `data.cutout` | mapping | | Cutout of the target. |
| `data.cutout.mode` | `fixed` or `auto` | `fixed` | `fixed` cuts `data.cutout.size` pixels about the target seed; `auto` cuts a preliminary region, measures where the radial profile of the target reaches the sky plateau, and cuts the plateau plus a margin. |
| `data.cutout.size` | integer | `null` | Cutout size in pixels for `fixed`. |
| `data.cutout.initial_factor` | number | 4.0 | `auto`: size of the preliminary region in units of the detected target extent. |
| `data.cutout.min_size` | integer | 100 for the preliminary region | `auto`: smallest cutout. |
| `data.cutout.max_size` | integer | none | `auto`: largest cutout. |
| `data.cutout.sky_margin` | number | 20.0 | `auto`: pixels added beyond the plateau radius. |
| `data.mask_path` | path | `null` | External FITS mask, `True` (non-zero) for excluded pixels, on the grid of the image. |
| `data.mask_mode` | `union`, `exact`, `replace` | `union` | `union` excludes the detected contaminants and the external mask; `exact` and `replace`, which are equivalent, use the external mask instead of the detected contaminants and require `data.mask_path`. |
| `data.pixel_scale` | number, `auto` or `null` | `null` | Pixel scale in arcsec. `null` or `auto` reads it from the FITS metadata. It is used for reporting, not by the fit. |

Rejected: `data.noise_map_path` and `data.psf_path`, which the
single-observation workflow does not support, and `data.external_mask_path`,
replaced by `data.mask_path`.

## `outputs`

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `outputs.root` | path | required | Root of the run directories, relative to the working directory. |
| `outputs.overwrite` | boolean | `false` | Allow the products of an existing run to be rebuilt. With `false`, completed preparation and PSF products are reused (AD-042) and a completed fit raises `FileExistsError`. |

## `preparation`

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `preparation.detection.n_sigma` | number | 3.0 | Detection threshold above the background, in units of its noise. |
| `preparation.detection.n_levels` | integer | 32 | Deblending levels of the segmentation. |
| `preparation.detection.n_pixels` | integer | 20 | Minimum number of connected pixels of a source. |
| `preparation.masking.negative_threshold_sigma` | number or `null` | 5.0 | Pixels below `−k` times the noise scale after background subtraction leave the likelihood support (AD-043); `null` disables the criterion. |
| `preparation.background.box_size` | integer | 30 | Inert: kept only because the section enters the preparation identity. |
| `preparation.background.filter_size` | integer | 3 | Inert, as above. |
| `preparation.noise` | mapping | `{}` | Reserved; a `method` entry is rejected. |

`preparation.detection.npixels` (now `n_pixels`), a top-level `masking`
section, and a `method` under `preparation.background` or
`preparation.noise` are rejected.

## `psf`

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `psf.type` | `empirical` or `gaussian` | `empirical` | An empirical PSF is built from stars of the image; a Gaussian PSF is analytic. |
| `psf.gaussian.fwhm` | number | | FWHM in pixels of a Gaussian PSF. |
| `psf.empirical.candidate_selection.auto` | boolean | `true` | Not read by the current code. |
| `psf.empirical.candidate_selection.target_n_candidates` | integer | 30 | Largest number of candidates kept. |
| `psf.empirical.candidate_selection.min_area` | integer | 10 | Smallest segment area of a candidate, in pixels. |
| `psf.empirical.candidate_selection.max_area` | integer | 800 | Largest segment area of a candidate, in pixels. |
| `psf.empirical.candidate_selection.min_peak` | number | 20 | Lowest peak of a candidate, in image units. |
| `psf.empirical.candidate_selection.max_peak` | number | 80000 | Highest peak of a candidate, in image units; not scaled to the image, and `auto` is rejected. |
| `psf.empirical.candidate_selection.min_peak_snr` | number | 30.0 | Lowest peak signal-to-noise ratio of a star used to build the PSF. |
| `psf.empirical.candidate_selection.max_eccentricity` | number | 0.8 | Largest eccentricity. |
| `psf.empirical.candidate_selection.max_elongation` | number | 2.5 | Largest elongation. |
| `psf.empirical.candidate_selection.edge_margin` | integer | 30 | Distance from the image edge, in pixels. |
| `psf.empirical.candidate_selection.galaxy_exclusion_box` | mapping or `null` | `null` | Box in which candidates are rejected. |
| `psf.empirical.candidate_selection.sigma_factor` | number | 4.0 | Factor converting the catalogue width of a candidate into its stamp half-size. |
| `psf.empirical.candidate_selection.size_margin` | integer | 4 | Pixels added to the stamp half-size. |
| `psf.empirical.candidate_selection.min_half_size` | integer | 12 | Smallest stamp half-size. |
| `psf.empirical.candidate_selection.max_half_size` | integer | 30 | Largest stamp half-size. |
| `psf.empirical.candidate_selection.isolation_margin` | integer | 2 | Extra margin, in pixels, of the isolation check. |
| `psf.empirical.construction.centroid_tolerance_pixels` | number | 0.05 | Tolerance, in pixels, of the validation of the kernel centroid. |
| `psf.empirical.construction.sky_inner_radius` | number | 80.0 | Inner radius, in pixels, of the ring where the sky of each star is measured (AD-046). |
| `psf.empirical.construction.sky_outer_radius` | number | 160.0 | Outer radius of that ring. |
| `psf.empirical.construction.mask_growth_pixels` | integer | 5 | Growth of the masks of other detected sources. |
| `psf.empirical.construction.support_half_size` | integer | 64 | Half-size of the kernel, which continues the stars' measured wings. It must satisfy `support_half_size < sky_inner_radius < sky_outer_radius`. |

## `fit`

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `fit.models` | list | `[single_sersic]` | Registered morphology models to fit, each in its own fit directory. |
| `fit.likelihood.type` | string | `gaussian` | Image likelihood; see below. |
| `fit.sampler.name` | `nautilus`, `dynesty`, `pocomc` | `nautilus` | Inference backend. |
| `fit.comparison.enabled` | boolean | `false` | Compare models by Bayes factor; requires `fit.models` to be exactly `single_sersic` and `double_sersic`. |
| `fit.comparison.metric` | string | `bayes_factor` | Comparison metric. |
| `fit.background.model` | `constant` or `plane` | `constant` | Residual background fitted with the morphology (AD-031). |
| `fit.background.priors` | mapping | | Priors of the background parameters. |
| `fit.priors` | mapping | | Priors of the morphology parameters, by model. |

### Likelihood

| Type | Options | Note |
| --- | --- | --- |
| `gaussian` | `include_normalization` (default `true`) | Independent pixels with the prepared noise scale. |
| `correlated_gaussian` | `include_normalization`, `factorization` (`type: nearest_neighbor`, `max_neighbors` 64 or 96) | Correlated noise of AD-047 through a nearest-neighbour factorization: the prepared sky scale and the stationary sky correlation, estimated on the whole sky region with one plane removed, no taper and lags up to 8. The estimator is fixed, so `calibration` options are rejected. A single-observation fit cannot yet use it (finding A1 of the deep audit), and the source variance of AD-047 is not yet implemented. |
| `cash` | `include_data_term`, `require_integer_data`, `integer_tolerance` | Poisson likelihood of images in counts. |
| `poisson_gaussian` | `read_noise`, `tail_probability`, `minimum_terms`, `maximum_terms` | Marginalized Poisson-Gaussian likelihood of images in counts. |

Every fit requires a completely normalized likelihood and is rejected before
any product is written otherwise: `include_normalization: false` or a Cash
likelihood without its data term is refused.

### Sampler

```yaml
fit:
  sampler:
    name: nautilus          # nautilus, dynesty or pocomc
    mode: static            # dynesty only: static or dynamic
    execution:
      processes: auto       # auto or a positive integer
      progress: true
    init: {}                # native keyword arguments of the sampler constructor
    run: {}                 # native keyword arguments of its run method
```

| Key | Default | Meaning |
| --- | --- | --- |
| `fit.sampler.mode` | `static` | Dynesty only: `static` uses `dynesty.NestedSampler`, `dynamic` uses `dynesty.DynamicNestedSampler`. |
| `fit.sampler.execution.processes` | `auto` | Worker processes; `auto` uses the available CPUs minus two, at most ten. |
| `fit.sampler.execution.progress` | `true` | Show the sampler's progress. |
| `fit.sampler.init` | `{}` | Passed unchanged to the sampler constructor, with its native names. |
| `fit.sampler.run` | `{}` | Passed unchanged to the sampler's run method. |

BayMorph keeps the native names of each sampler and does not translate them;
see the documentation of the installed Nautilus, Dynesty or pocoMC version.
The likelihood, prior transform, parameter order and dimension belong to
BayMorph and cannot be overridden there. `init.filepath` for Nautilus and
`run.resume_state_path` for pocoMC accept `auto`, a signed checkpoint inside
the run directory.

### Priors

Every sampled parameter needs a typed prior. A prior is a mapping with exactly
the keys of its type:

| Type | Keys |
| --- | --- |
| `uniform` | `lower`, `upper`, optionally `relative_to: target_x` or `target_y` |
| `log_uniform` | `lower`, `upper` |
| `normal` | `mean`, `sigma` |
| `beta` | `alpha`, `beta`, and optionally both `lower` and `upper` |
| `fixed` | `value`; accepted by the parser, not yet validated in the single-observation production path |

Angles may be written as expressions of `pi`, such as `-pi/2`. A prior with
`relative_to` is centred on the target seed transformed to cutout
coordinates, which is a deterministic change of coordinates, not information
taken from the pixels.

The defaults are uniform priors in pixels. No model has a default prior for
its fluxes, which must always be declared.

| Model | Parameters | Default priors |
| --- | --- | --- |
| `single_sersic` | `flux`, `re`, `n`, `q`, `theta`, `x0`, `y0` | `re` 1–120, `n` 0.5–6, `q` 0.4–1, `theta` −π/2–π/2, `x0` and `y0` ±8 about the target seed |
| `double_sersic` | `flux_bulge`, `re_bulge`, `n_bulge`, `q_bulge`, `theta_bulge`, `flux_disk`, `re_disk`, `n_disk`, `q_disk`, `theta_disk`, `x0`, `y0` | `re_bulge` 1–30, `n_bulge` 1.5–6, `re_disk` 10–120, `n_disk` 0.5–3, `q` 0.4–1, `theta` −π/2–π/2, centre ±8 |
| `sersic_point_source` | `flux_sersic`, `re`, `n`, `q`, `theta`, `flux_point_source`, `x0`, `y0` | as `single_sersic` |
| `bulge_disk` | `flux_bulge`, `re_bulge`, `n_bulge`, `q_bulge`, `theta_bulge`, `flux_disk`, `re_disk`, `q_disk`, `theta_disk`, `x0`, `y0` | `re_bulge` 1–30, `n_bulge` 1.5–6, `q_bulge` and `q_disk` 0.4–1, `theta_bulge` and `theta_disk` −π/2–π/2, `re_disk` 10–120, centre ±8 |
| `boxy_sersic` | `flux`, `re`, `n`, `q`, `c0`, `theta`, `x0`, `y0` | as `single_sersic`, and `c0` −1–1 |
| `core_sersic` | `flux`, `re`, `n`, `rb`, `gamma`, `alpha`, `q`, `theta`, `x0`, `y0` | as `single_sersic`, and `rb` 0.5–20, `gamma` 0–0.9, `alpha` 0.5–2 |
| `bulge_broken_disk` | `flux_bulge`, `re_bulge`, `n_bulge`, `q_bulge`, `theta_bulge`, `flux_disk`, `h_in_disk`, `h_out_disk`, `rb_disk`, `wb_disk`, `q_disk`, `theta_disk`, `x0`, `y0` | the bulge as `bulge_disk`; `h_in_disk` and `h_out_disk` 2–80, `rb_disk` 5–300, `wb_disk` 0.5–20, `q_disk` 0.4–1, `theta_disk` −π/2–π/2, centre ±8 |
| `bulge_disk_bar` | `flux_bulge`, `re_bulge`, `n_bulge`, `q_bulge`, `theta_bulge`, `flux_disk`, `re_disk`, `q_disk`, `theta_disk`, `flux_bar`, `a_bar`, `alpha_bar`, `c0_bar`, `q_bar`, `theta_bar`, `x0`, `y0` | the bulge and the disc as `bulge_disk`; `a_bar` 5–100, `alpha_bar` 2–5, `c0_bar` 0–3.5, `q_bar` 0.2–0.7, `theta_bar` −π/2–π/2, centre ±8 |
| `bulge_disk_ring` | `flux_bulge`, `re_bulge`, `n_bulge`, `q_bulge`, `theta_bulge`, `flux_disk`, `re_disk`, `q_disk`, `theta_disk`, `flux_ring`, `r_ring`, `sigma_ring`, `q_ring`, `theta_ring`, `x0`, `y0` | the bulge and the disc as `bulge_disk`; `r_ring` 5–200, `sigma_ring` 2–20, `q_ring` 0.5–1, `theta_ring` −π/2–π/2, centre ±8 |
| `bulge_disk_spiral` | `flux_bulge`, `re_bulge`, `n_bulge`, `q_bulge`, `theta_bulge`, `flux_disk`, `re_disk`, `q_disk`, `theta_disk`, `amplitude_spiral`, `winding_spiral`, `phase_spiral`, `r_in_spiral`, `x0`, `y0` | the bulge and the disc as `bulge_disk`; `amplitude_spiral` 0–1, `winding_spiral` −6–6, `phase_spiral` 0–π, `r_in_spiral` 1–60, centre ±8 |

The models of AD-048 are validated for single-observation fits within a
rendering domain:

| Model | Validated domain |
| --- | --- |
| `boxy_sersic` | `c0` −1–1, `n` up to 6, `re` at least 1 |
| `core_sersic` | `n` up to 6, `re` at least 1, `rb` at least 0.5, `gamma` 0–0.9, `alpha` 0.5–2 |
| `bulge_broken_disk` | `wb_disk` at least 0.5 |
| `bulge_disk_bar` | `a_bar` at least 5, `alpha_bar` at least 2, `c0_bar` 0–3.5, `q_bar` at least 0.2 |
| `bulge_disk_ring` | `sigma_ring` at least 2, `q_ring` at least 0.5 |
| `bulge_disk_spiral` | `winding_spiral` −6–6, `re_disk` at least 10, `q_disk` at least 0.4, `r_in_spiral` at least 1 |

A prior whose support leaves the domain is rejected, and so is a joint
multi-observation fit with any of these models. Within the domain, the
rendering of a bright bar, or of a compact bright bulge, can still reach the
noise of a very deep image; the page of each model states where.

For `double_sersic` and `bulge_disk`, a prior declared under `q` or `theta`
applies to both components (`q_bulge` and `q_disk`, `theta_bulge` and
`theta_disk`) unless a component-specific prior is declared. The two
components remain separate parameters.

The background priors are declared under `fit.background.priors`:

| Key | Default |
| --- | --- |
| `fit.background.priors.background_level` | uniform −50 to 50, in image units, not scaled to the image |
| `fit.background.priors.background_gradient_x` | uniform −0.1 to 0.1 per pixel, used by `plane` |
| `fit.background.priors.background_gradient_y` | uniform −0.1 to 0.1 per pixel, used by `plane` |

A `sky` prior under `fit.priors.<model>` is rejected: the residual background
belongs to `fit.background` (AD-044).

## `diagnostics`

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `diagnostics.format` | string | `pdf` | File format of the figures. |
| `diagnostics.corner` | boolean | `true` | Corner plot of the weighted posterior. |
| `diagnostics.residual` | boolean | `true` | Data, model and residual images. |
| `diagnostics.radial_profile` | boolean | `true` | Radial profile of data and model. |
| `diagnostics.posterior_profile_bands` | boolean | `true` | Posterior percentile bands on the radial profile. |
| `diagnostics.titles` | mapping | none | Figure title by model name. |

## `simulation`

Present only in a configuration that generates its own synthetic image, as
the example `simulated_galaxy.yaml` does. `BayMorphRunner.simulate()` and
`baymorph run` write the image to `data.image_path` and the truth to the
`truth/` directory of the run.

| Key | Meaning |
| --- | --- |
| `simulation.shape` | Image shape `[ny, nx]`. |
| `simulation.seed` | Random seed of the noise. |
| `simulation.truth` | True parameters: `flux`, `re`, `n`, `q`, `theta_deg`, `x0`, `y0`. |
| `simulation.psf` | PSF of the simulation, for example `type: gaussian` and `fwhm`. |
| `simulation.noise` | `sky` level, `poisson_noise` on or off, and `read_noise`. |
