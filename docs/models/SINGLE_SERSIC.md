# Single Sérsic (`single_sersic`)

The baseline morphology of BayMorph: the light of the whole galaxy described
by one elliptical [Sérsic profile](COMPONENTS.md#sersic-profile), the law of
Sérsic ([1963][sersic1963], [1968][sersic1968]).

## Definition

$$
I(\mathbf{x}) = I_{\mathrm{Sersic}}(\mathbf{x};\,
F, R_e, n, q, \theta, x_0, y_0).
$$

$b_n$ solves $\gamma(2n, b_n) = \Gamma(2n)/2$, so that $R_e$ holds half the
light ([Ciotti 1991][ciotti1991]), and `flux` is the total flux of the
profile over the whole plane
([Graham & Driver 2005][graham-driver2005]); the
[Components](COMPONENTS.md#sersic-profile) page gives both.

![Two plots: surface brightness relative to its value at Re against R/Re for n from 0.5 to 6, and the fraction of light enclosed within R](figures/sersic_profiles-light.png#gh-light-mode-only)
![Two plots: surface brightness relative to its value at Re against R/Re for n from 0.5 to 6, and the fraction of light enclosed within R](figures/sersic_profiles-dark.png#gh-dark-mode-only)

*Left: the profile for indices from 0.5 to 6; a higher index means a steeper core and more extended wings. Right: the light enclosed within R, which is one half at R = Rₑ for every index.*

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
  real image the local correlations were 0.99 between `flux` and `re`, 0.95
  between `flux` and `n`, and 0.98 between `re` and `n`.
  Assess them jointly, for example with
  `InferenceResult.mahalanobis_distance`.
- The residual background is anticorrelated with the structural parameters,
  because the outer profile and the background both add light far from the
  centre.
- `theta` is undefined when `q` approaches 1, and has period π.
- `n` and `re` follow the PSF core: each percent of error in its width moves
  `n` by about 1 percent and `re` by about 2 percent.

![Two plots: the radial profiles of a Sérsic of index 4 and of the index-3 Sérsic that best matches it, and their difference in percent against radius](figures/sersic_degeneracy-light.png#gh-light-mode-only)
![Two plots: the radial profiles of a Sérsic of index 4 and of the index-3 Sérsic that best matches it, and their difference in percent against radius](figures/sersic_degeneracy-dark.png#gh-dark-mode-only)

*A Sérsic of index 4 (Rₑ = 10 px, F = 100) and the index-3 Sérsic that matches its convolved image best by least squares: F = 83 and Rₑ = 7.1 px. A lower index is compensated by a lower flux and a smaller radius within a few percent inside 10 px; the profiles part in the faint wings, where the noise and the background decide.*

## Validation

- **Rendering.** The pixel integrals agree with an independent reference
  within 6e-9 of the total flux. With an empirical PSF, the effective-PSF
  rendering recovers every parameter of simulated galaxies within 0.09
  standard deviations of the truth.
- **Very deep images.** Under an empirical PSF the rendering error is a
  fixed fraction of the profile, so it becomes comparable with the noise
  above a peak signal-to-noise that depends on n and Rₑ: from 1.2e3 for
  n = 6 and Rₑ = 1 px to about 2e4 for shallower profiles.
- **Calibration.** On fifty noise realizations of one simulated galaxy,
  with a Gaussian PSF and Gaussian noise, the posteriors were found
  calibrated. That campaign used the likelihood of independent pixels that
  preceded the current noise model, and it has not been repeated with it.
- **Noise model.** On simulated coadds whose noise has a Poisson term from
  the galaxy and a correlated sky, a model of independent pixels reported
  intervals up to three times too narrow; the noise model of BayMorph brings
  the ratio of the true to the reported width to between 0.90 and 1.16.
- **Real data.** No fit of a real image has been validated yet.

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

## References

- [Ciotti L., 1991, A&A, 249, 99][ciotti1991]: the equation that defines $b_n$.
- [Graham A. W., Driver S. P., 2005, PASA, 22, 118][graham-driver2005]: the total flux and the other quantities of the profile.
- [Sérsic J. L., 1963, Boletín de la Asociación Argentina de Astronomía, 6, 41][sersic1963]: the profile.
- [Sérsic J. L., 1968, Atlas de Galaxias Australes, Observatorio Astronómico, Córdoba][sersic1968]: the profile.

[ciotti1991]: https://ui.adsabs.harvard.edu/abs/1991A%26A...249...99C/abstract
[graham-driver2005]: https://ui.adsabs.harvard.edu/abs/2005PASA...22..118G/abstract
[sersic1963]: https://ui.adsabs.harvard.edu/abs/1963BAAA....6...41S/abstract
[sersic1968]: https://ui.adsabs.harvard.edu/abs/1968adga.book.....S/abstract
