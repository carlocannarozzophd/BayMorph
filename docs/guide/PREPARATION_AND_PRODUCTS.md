# Preparation and Products

This page describes what BayMorph does to an image before fitting it and the
products it writes. Every product of an analysis lives under
`outputs.root/object.name/`.

## Image preparation

`prepare_image()`, or the first stage of `baymorph run`, turns the input
image into the data, noise and support of the likelihood:

1. **Detection.** Sources are segmented with Photutils
   (`preparation.detection`). The segment at the target seed is the target;
   the other segments are contaminants.
2. **Cutout.** The target is cut out, at a fixed size or, with
   `data.cutout.mode: auto`, out to where its radial profile reaches the sky
   plus a margin.
3. **Fitting mask.** The contaminants and an optional external mask are
   excluded, as `data.mask_mode` declares.
4. **Sky region.** The elliptical radial profile of the target is searched
   for the radius where it flattens, the plateau. Everything beyond that
   radius plus 5 px, minus the masked sources, is the sky region.
5. **Background.** The median of the clipped sky region is subtracted as a
   constant.
6. **Noise.** The noise scale is measured on the sky region (see below) and
   written as a constant noise map.
7. **Support.** Pixels below `−k` times the noise scale, with
   `k = preparation.masking.negative_threshold_sigma`, leave the likelihood
   support (AD-043). Source detection thresholds only positive excursions,
   so a negative detector artifact needs this separate criterion; it reads
   the prepared data, never a fitted model.

The products are written to `image_preparation/`: `prepared_image.fits`,
`noise_map.fits`, `mask.fits` and `fitting_mask.fits`, with the segmentation,
the radial profile, the sky region and the metadata. A completion marker is
written last. A later run with the same preparation identity reuses them
(AD-042).

### The prepared noise scale

The noise scale is a truncation-corrected Gaussian scale: sky pixels are
iteratively sigma-clipped, the standard deviation of the survivors is
computed, and the truncation caused by the clipping is inverted to recover
the parent scale (AD-040). The scaled median absolute deviation it replaced
was not biased but inefficient: its run-to-run scatter was about 1.1 percent
on the sky region of the example, against 0.7 percent now, and the scale
enters the likelihood normalization, so it sets every interval width and the
absolute evidence.

The scale is measured about a plane fitted to the sky region, not about a
constant, so an unmodelled sky gradient is not counted as noise (AD-041).
Preparation still subtracts a constant; the plane only measures the scale.
An unmodelled gradient therefore shows up as an elevated reduced chi-square
rather than being hidden. The remedy is `fit.background.model: plane`, which
fits `background_gradient_x` and `background_gradient_y`. The fitted
gradient, its standard error and its significance per axis are reported in
the preparation metadata and in the runtime report, under
`image_preparation.background.sky_gradient`. They are diagnostics: the
choice of background model remains the user's.

Limitations, all explicit:

- the correction assumes that the surviving core is Gaussian;
- only a linear gradient is removed, so higher-order sky structure is
  counted as noise;
- the prepared sky scale is the marginal scale of the noise model of
  [AD-047](../research/NOISE_MODEL_PROPOSAL_2026-10-02.md), which adds the
  Poisson variance of the source, with a declared effective gain, and the
  correlation of the sky, estimated on the sky region after preparation; an
  external variance map is not yet supported.

`baymorph.stats.truncated_gaussian_sigma` and `truncated_variance_ratio`
expose the correction. The separate `estimate_noise_map` serves only the
observation advisor and never sets the likelihood noise.

## Empirical PSF

With `psf.type: empirical`, `select_psf_candidates()` writes the candidate
stars to `psf/psf_candidates.json`. `build_psf()` then builds the kernel with
the builder of AD-046 and writes `psf/psf.fits`: the native kernel and, in
its `SUBPIXEL` extension, the kernel at the eleven sub-pixel phases of each
axis, through which the forward model renders the profile, with
`psf/psf_metadata.json`. An empirical PSF built before AD-046 has no
sub-pixel kernels and is rejected for fits. [Public API](../API.md)
describes the builder.

## Noise products

The noise model of AD-047 needs two products for each observation, which
`fit()`, or the fit stage of `baymorph run`, builds before the sampler
starts and writes beside the preparation products:

| Product | Content |
| --- | --- |
| `image_preparation/correlated_gaussian_noise.npz` | The stationary correlation of the sky up to lag 8, estimated on the sky region with one plane removed and without a taper, with its metadata: the number of sky pixels, the smallest number of pixel pairs at a lag, the smallest value of its spectrum, the correlation along each axis up to lag 16, whose lags 9 to 16 are a diagnostic of the truncation, and the fingerprints of the preparation products it was estimated from. |
| `image_preparation/correlated_gaussian_vecchia/m24/` | The nearest-neighbour factorization of that correlation on the likelihood support, with 24 neighbours, and its `manifest.json`. |

Both are immutable: a later fit of the same preparation reuses them. A sky
region with fewer than 1000 pixel pairs at some lag up to 16, or a
correlation whose spectrum is not strictly positive, stops the fit instead
of being repaired. Products of the schemas before AD-047 are rejected, with
a message that asks for a new output namespace. On small cutouts the
estimate of the correlation is noisy, and its uncertainty is not
propagated; the packaged simulated example is 201 x 201 px for that
reason.

The effective gain is resolved from the source that
`fit.likelihood.effective_gain` declares ([Configuration](CONFIGURATION.md)).
`fit_summary.json` records it, with the estimator of the correlation and the
paths of both products, under `likelihood`.

## Fit products

Each model of `fit.models` has its own directory `fits/<model>/`:

| File | Content |
| --- | --- |
| `posterior.npz` | Weighted posterior samples, the statistical source of truth. |
| `best_parameters.json` | Best-fitting parameters. |
| `fit_summary.json`, `fit_summary.txt` | Evidence, goodness of fit, sampler configuration and provenance. |
| `best_model_image.npy`, `residual.npy`, `norm_residual.npy`, `noise.npy` | Model image; residual; residual divided by the scale of the noise model at the best sample, which includes the source variance when a gain is declared, a marginal standardization that is not whitened; and the prepared sky scale. |
| `config_resolved.yaml` | Resolved configuration of the fit. |
| `corner.<format>`, `data_model_residual.<format>`, `radial_profile.<format>` | Diagnostics, as `diagnostics` configures. |
| `COMPLETE.json` | Completion marker, written last, after every product validates. |

With `fit.comparison.enabled`, `model_comparison/bayes_factor.json` holds the
Bayes factor. The run directory also holds the resolved configuration and
`runtime_report.yaml`.

Equal-weight samples are derived from the weighted samples for plotting
only: summaries, intervals and evidence always use the weights.

## Simulated runs

A configuration with a `simulation` section writes the synthetic image to
`data.image_path`, usually under `input/`, and its truth to
`truth/truth.json`. The corner plot of such a run marks the truth, detected
by the presence of that file, so real-data fits never receive truth markers.
The true `x0` and `y0` are full-image coordinates and are shifted by the
origin of the cutout, because the sampled positions are cutout coordinates.
The simulated sky is never mapped onto `background_level`: preparation
removes the sky, so the sampled background is a residual of the prepared
image. Joint corner plots carry no truth markers, since each observation has
its own cutout origin.

`baymorph.plotting.corner_plot` accepts `truths`, ordered as its
`parameter_labels`, with `truth_color` and `truth_linewidth`; entries that
are `None` or NaN leave their parameter unmarked. Truth markers are an
overlay: they do not enter any summary, interval or evidence, and they do
not extend the axes, so a truth outside the plotted range is clipped.

## Joint runs

A multi-observation analysis keeps the preparation and PSF products in the
namespace of each observation, and writes the joint posterior and the
per-observation model and residual images under its own namespace, as
[Multi-Observation Workflow](../architecture/MULTI_OBSERVATION_PUBLIC_WORKFLOW.md)
specifies.
