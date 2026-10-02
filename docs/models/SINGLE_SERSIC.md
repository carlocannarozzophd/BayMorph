# Single Sérsic (`single_sersic`)

The baseline morphology of BayMorph: the light of the whole galaxy described
by one elliptical [Sérsic profile](COMPONENTS.md#sersic-profile).

## Definition

$$
I(\mathbf{x}) = I_{\mathrm{Sersic}}(\mathbf{x};\,
F, R_e, n, q, \theta, x_0, y_0).
$$

## Parameters

| Parameter | Meaning | Units | Default prior |
| --- | --- | --- | --- |
| `flux` | total flux of the profile | image units | none: must be declared |
| `re` | half-light radius along the major axis | pixels | uniform 1 to 120 |
| `n` | Sérsic index | | uniform 0.5 to 6 |
| `q` | axis ratio | | uniform 0.4 to 1 |
| `theta` | position angle | radians | uniform −π/2 to π/2 |
| `x0`, `y0` | centre | pixels | uniform ±8 about the target seed |

The default priors are in pixels and must be checked against the size of
the target and the resolution of the image; they are scientific assumptions,
not sampler settings.

![Grid of sixteen convolved Sérsic images: each row varies one parameter, the index n, the radius Re, the axis ratio q and the position angle theta](figures/single_sersic_parameters-light.png#gh-light-mode-only)
![Grid of sixteen convolved Sérsic images: each row varies one parameter, the index n, the radius Re, the axis ratio q and the position angle theta](figures/single_sersic_parameters-dark.png#gh-dark-mode-only)

*What each parameter does, one row at a time, with the others fixed at n = 2, Rₑ = 7 px, q = 0.7 and θ = 30°, after a Gaussian PSF of 2.5 px FWHM.*

## What it represents

One smooth, elliptical light distribution whose ellipticity and position
angle do not change with radius. It does not represent a separate nucleus,
bar, disc, ring or spiral structure: their light is absorbed into the one
profile, which then describes none of them. A profile misfit appears in the
residuals and in the reduced chi-square, not in the posterior widths.

## Known degeneracies

- The flux, the effective radius and the index are strongly correlated: on a
  real g-band problem the local correlations were 0.99 between `flux` and
  `re`, 0.95 between `flux` and `n`, and 0.98 between `re` and `n`
  ([Inference Geometry](../architecture/INFERENCE_GEOMETRY.md)). Assess them
  jointly, for example with `InferenceResult.mahalanobis_distance`.
- The residual background is anticorrelated with the structural parameters,
  because the outer profile and the background both add light far from the
  centre.
- `theta` is undefined when `q` approaches 1, and has period π.
- `n` and `re` follow the PSF core: each percent of error in its width moves
  `n` by about 1 percent and `re` by about 2 percent
  ([PSF parameter sensitivity](../research/PSF_PARAMETER_SENSITIVITY_2026-10-01.md)).

![Two plots: the radial profiles of a Sérsic of index 4 and of the index-3 Sérsic that best matches it, and their difference in percent against radius](figures/sersic_degeneracy-light.png#gh-light-mode-only)
![Two plots: the radial profiles of a Sérsic of index 4 and of the index-3 Sérsic that best matches it, and their difference in percent against radius](figures/sersic_degeneracy-dark.png#gh-dark-mode-only)

*A Sérsic of index 4 (Rₑ = 10 px, F = 100) and the index-3 Sérsic that matches its convolved image best by least squares: F = 83 and Rₑ = 7.1 px. A lower index is compensated by a lower flux and a smaller radius within a few percent inside 10 px; the profiles part in the faint wings, where the noise and the background decide.*

## Validation

- **Rendering.** The pixel integrals agree with an independent reference
  within 6e-9 of the total flux; with an empirical PSF, the effective-PSF
  rendering of AD-046 recovers every parameter of galaxies like the target
  within 0.09 statistical deviations of the truth.
- **Calibration on simulations.** A repeated-noise campaign of fifty
  realizations on 2026-09-19 found the posteriors calibrated on simulated
  images with a Gaussian PSF and Gaussian noise
  (`validation/repeated_noise_recovery/`). Its summary is not yet recorded in
  a research note.
- **Noise model.** On simulated coadds whose noise has a source Poisson term
  and a correlated sky, the production noise model reports intervals up to
  three times too narrow
  ([noise-model bench](../research/NOISE_MODEL_BENCH_2026-10-01.md)); the
  proposed AD-047 addresses it.
- **Real data.** No real-data fit is validated yet. The completed r-band fit
  of the production target is a demonstration of the workflow.

## Configuration

```yaml
fit:
  models:
    - single_sersic
  priors:
    single_sersic:
      flux:
        type: uniform
        lower: 1000.0
        upper: 100000.0
      re:
        type: uniform
        lower: 2.0
        upper: 40.0
  background:
    model: constant
    priors:
      background_level:
        type: uniform
        lower: -10.0
        upper: 10.0
```

Priors not declared take the defaults above; the values here are an
example, to be chosen for each image.

## Python

`baymorph.SersicModel` is the intrinsic profile;
`baymorph.fitting.make_single_sersic_psf_model` builds the PSF-convolved
model a fit evaluates.
