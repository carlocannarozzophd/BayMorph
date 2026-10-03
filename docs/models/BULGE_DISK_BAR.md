# Bulge, Disc and Bar (`bulge_disk_bar`)

A Sérsic bulge, an exponential disc and a Ferrers bar
([Ferrers bar](COMPONENTS.md#ferrers-bar)), sharing a centre. Available for
fits of a single image.

## Definition

$$
I(\mathbf{x}) = I_{\mathrm{Sersic}}(\mathbf{x};\,F_b, R_{e,b}, n_b, q_b,
\theta_b) + I_{\mathrm{exp}}(\mathbf{x};\,F_d, R_{e,d}, q_d, \theta_d)
+ I_{\mathrm{bar}}(\mathbf{x};\,F_{\mathrm{bar}}, a, \alpha, c_0,
q_{\mathrm{bar}}, \theta_{\mathrm{bar}}),
$$

the bulge and the disc of [`bulge_disk`](BULGE_DISK.md) and a bar whose
light ends at the semi-major length $a$,

$$
I_{\mathrm{bar}}(m) = I_0 \left[1 - \left(\frac{m}{a}\right)^{2}
\right]^{\alpha} \quad (m < a), \qquad I_{\mathrm{bar}}(m) = 0
\quad (m \ge a),
$$

on generalized ellipses $m = (|x'|^{c} + |y'/q_{\mathrm{bar}}|^{c})^{1/c}$,
$c = 2 + c_0$. This is the Ferrers profile of GALFIT
([Peng et al. 2010][peng2010]) with its central slope fixed to zero, named
after the ellipsoids of [Ferrers (1877)][ferrers1877]. Boxy isophotes,
$c_0 > 0$, describe the rectangular isophotes of bars
([Athanassoula et al. 1990][athanassoula1990]). The flux of the bar is
$F_{\mathrm{bar}} = K(c)\, q_{\mathrm{bar}}
I_0\, a^2/(\alpha + 1)$, with $K(c)$ the area factor of the generalized
ellipse.

## Parameters

| Parameter | Meaning | Units | Default prior |
| --- | --- | --- | --- |
| `flux_bulge` | total flux of the bulge | image units | none: must be declared |
| `re_bulge` | half-light radius of the bulge | pixels | uniform 1 to 30 |
| `n_bulge` | Sérsic index of the bulge | | uniform 1.5 to 6 |
| `q_bulge` | axis ratio of the bulge | | uniform 0.4 to 1 |
| `theta_bulge` | position angle of the bulge | radians | uniform −π/2 to π/2 |
| `flux_disk` | total flux of the disc | image units | none: must be declared |
| `re_disk` | half-light radius of the disc | pixels | uniform 10 to 120 |
| `q_disk` | axis ratio of the disc | | uniform 0.4 to 1 |
| `theta_disk` | position angle of the disc | radians | uniform −π/2 to π/2 |
| `flux_bar` | total flux of the bar | image units | none: must be declared |
| `a_bar` | semi-major length of the bar | pixels | uniform 5 to 100 |
| `alpha_bar` | sharpness of the edge of the bar | | uniform 2 to 5 |
| `c0_bar` | isophote shape of the bar | | uniform 0 to 3.5 |
| `q_bar` | axis ratio of the bar | | uniform 0.2 to 0.7 |
| `theta_bar` | position angle of the bar | radians | uniform −π/2 to π/2 |
| `x0`, `y0` | shared centre | pixels | uniform ±8 about the target seed |

The default priors are generic and in pixels; they must be checked against
the target and the image. `a_bar` must be at least 5 px, `alpha_bar` at
least 2, `c0_bar` between 0 and 3.5 and `q_bar` at least 0.2: the domain in
which the rendering of the bar is validated, which configuration validation
enforces. Flatter or thinner bars are integrated accurately over pixels, but
their edge, sampled at pixel centres outside the window of the effective-PSF
rendering, is not.

![Images of an elliptical and a boxy bar with their isophotes, and a plot of the profile of a Ferrers bar against radius for four sharpnesses of its edge](figures/ferrers_bar_profiles-light.png#gh-light-mode-only)
![Images of an elliptical and a boxy bar with their isophotes, and a plot of the profile of a Ferrers bar against radius for four sharpnesses of its edge](figures/ferrers_bar_profiles-dark.png#gh-dark-mode-only)

*Top: bars with a = 30 px, α = 2 and q = 0.35, on ellipses and on boxy isophotes, with their isophotes at m = 10, 20 and 29 px. Bottom: the profile of the bar along its major axis for four values of α; the slope at the edge is infinite for α < 1.*

![Images of the convolved bulge, disc, bar and their sum, with the radial profiles of the four](figures/bulge_disk_bar_decomposition-light.png#gh-light-mode-only)
![Images of the convolved bulge, disc, bar and their sum, with the radial profiles of the four](figures/bulge_disk_bar_decomposition-dark.png#gh-dark-mode-only)

*A bulge of 20 percent of the light (n = 3, Rₑ = 3 px), a disc of 55 percent (Rₑ = 18 px) and a bar of 25 percent (a = 20 px, α = 2, c₀ = 1.5, q = 0.3), after a Gaussian PSF of 2.5 px FWHM. The profiles are averages over circles, so the edge of the inclined bar appears smoothed.*

## What it represents

A barred disc galaxy with a bulge. Ferrers profiles describe bars, and
lenses with a flat profile, in multicomponent decompositions of S0 and
early-type spiral galaxies
([Laurikainen, Salo & Buta 2005][laurikainen2005]). A bar left out of a
decomposition can make the bulge-to-total ratio about twice too large,
because the bulge absorbs the light of the bar
([Gadotti 2008][gadotti2008]).

## Known degeneracies

- The bar and the bulge share the light at small radii, the more so for a
  short bar or a round bulge (Gadotti 2008).
- `a_bar` and `alpha_bar` are correlated: a shorter bar with a soft edge and
  a longer one with a sharp edge can look alike once convolved with the PSF.
- `c0_bar` and `q_bar` both shape the isophotes of a thin bar, and are
  correlated when the bar is barely resolved across its minor axis.
- The bar and the disc share the light along the bar, so `flux_bar` and
  `flux_disk` are anticorrelated.

## Validation

- **Rendering.** Tested on 230 cases under PSFs of 2.3 and 3.5 px FWHM.
  Within the domain above, the largest pixel error is 2.2e-3 of the peak and
  the largest flux error 8.7e-4. Bars with a softer edge, `alpha_bar` below
  2, or thinner than `q_bar` = 0.2 are not rendered accurately at their edge
  ([details](https://github.com/carlocannarozzophd/BayMorph/blob/main/docs/research/MORPHOLOGY_FAMILIES_RENDERING_2026-10-02.md)).
- **Very bright bars.** The tips of the bar, more than 14 px from the
  centre, are rendered from samples at pixel centres. Under a PSF of
  2.3 px FWHM a bar whose own peak signal-to-noise exceeds about 1500 is
  therefore rendered with an error comparable with the noise.
- **Flux.** A bar inside the image holds its analytic flux within 1e-7, its
  edge included.
- **Recovery.** On a simulated barred galaxy, the posterior of `a_bar` and
  `q_bar` covers the true values.
- **Cost.** About two and a half times the cost of `bulge_disk`.
- **Real data.** Not yet used on real images.

## Configuration

```yaml
fit:
  models:
    - bulge_disk_bar
  priors:
    bulge_disk_bar:
      flux_bulge:
        type: uniform
        lower: 1000.0
        upper: 100000.0
      flux_disk:
        type: uniform
        lower: 1000.0
        upper: 200000.0
      flux_bar:
        type: uniform
        lower: 500.0
        upper: 50000.0
```

Priors not declared take the defaults above; the values here are an
example, to be chosen for each image. A joint fit of several images does not
accept this model.

## Python

`baymorph.models.FerrersBarComponent` is the bar;
`baymorph.fitting.make_bulge_disk_bar_psf_model` builds the PSF-convolved
model a fit evaluates.

## References

- [Athanassoula E. et al., 1990, MNRAS, 245, 130][athanassoula1990]: the boxy isophotes of bars.
- [Ferrers N. M., 1877, Q. J. Pure Appl. Math., 14, 1][ferrers1877]: the ellipsoids after which the Ferrers profile is named.
- [Freeman K. C., 1970, ApJ, 160, 811][freeman1970]: the exponential disc.
- [Gadotti D. A., 2008, MNRAS, 384, 420][gadotti2008]: the bias of an omitted bar.
- [Laurikainen E., Salo H., Buta R., 2005, MNRAS, 362, 1319][laurikainen2005]: Ferrers bars in decompositions of S0 galaxies.
- [Peng C. Y. et al., 2010, AJ, 139, 2097][peng2010]: the Ferrers profile of bars.
- [Sérsic J. L., 1968, Atlas de Galaxias Australes, Observatorio Astronómico, Córdoba][sersic1968]: the Sérsic profile of the bulge.

[athanassoula1990]: https://ui.adsabs.harvard.edu/abs/1990MNRAS.245..130A/abstract
[ferrers1877]: https://ui.adsabs.harvard.edu/abs/1877QJPAM..14....1F/abstract
[freeman1970]: https://ui.adsabs.harvard.edu/abs/1970ApJ...160..811F/abstract
[gadotti2008]: https://ui.adsabs.harvard.edu/abs/2008MNRAS.384..420G/abstract
[laurikainen2005]: https://ui.adsabs.harvard.edu/abs/2005MNRAS.362.1319L/abstract
[peng2010]: https://ui.adsabs.harvard.edu/abs/2010AJ....139.2097P/abstract
[sersic1968]: https://ui.adsabs.harvard.edu/abs/1968adga.book.....S/abstract
