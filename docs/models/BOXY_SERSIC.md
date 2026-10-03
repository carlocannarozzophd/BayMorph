# Boxy or Discy Sérsic (`boxy_sersic`)

A [Sérsic profile](COMPONENTS.md#sersic-profile) whose isophotes are
generalized ellipses, boxy or discy instead of elliptical
([generalized Sérsic](COMPONENTS.md#generalized-sersic-profile)).
Available for fits of a single image.

## Definition

$$
I(\mathbf{x}) = I_0 \exp\left[-b_n \left(\frac{m}{R_e}\right)^{1/n}\right],
\qquad
m = \left(|x'|^{c} + \left|\frac{y'}{q}\right|^{c}\right)^{1/c},
\qquad c = 2 + c_0,
$$

where $(x', y')$ are the offsets from the centre along the major and minor
axes. $c_0 = 0$ gives ellipses and the [`single_sersic`](SINGLE_SERSIC.md)
model; $c_0 > 0$ gives boxy and $c_0 < 0$ discy isophotes, the convention
of GALFIT ([Peng et al. 2010][peng2010]) and Imfit ([Erwin 2015][erwin2015])
for the generalized ellipses of
[Athanassoula et al. (1990)][athanassoula1990]. The area within the
isophote $m$ is $K(c)\,q\,m^2$ with
$K(c) = 4\,\Gamma(1 + 1/c)^2 / \Gamma(1 + 2/c)$, $\pi$ over the area ratio
$R(c)$ of [Peng et al. (2002)][peng2002], so
the fraction of light within an isophote depends on $m$ alone: $R_e$ is the
semi-major radius of the half-light isophote for every $c_0$, and the total
flux is

$$
F = 2\,K(c)\,q\,I_0 R_e^2\, n\, b_n^{-2n}\,\Gamma(2n).
$$

## Parameters

| Parameter | Meaning | Units | Default prior |
| --- | --- | --- | --- |
| `flux` | total flux of the profile | image units | none: must be declared |
| `re` | semi-major radius of the half-light isophote | pixels | uniform 1 to 120 |
| `n` | Sérsic index | | uniform 0.5 to 6 |
| `q` | axis ratio of the isophotes | | uniform 0.4 to 1 |
| `c0` | isophote shape: negative discy, positive boxy | | uniform −1 to 1 |
| `theta` | position angle of the major axis | radians | uniform −π/2 to π/2 |
| `x0`, `y0` | centre | pixels | uniform ±8 about the target seed |

The default priors are generic and in pixels; they must be checked against
the target and the image. `c0` must lie in the domain where the rendering is
validated, which configuration validation enforces.

![Three images of one Sérsic profile with discy, elliptical and boxy isophotes, with three isophotes drawn on each, and the outer isophote of each drawn together](figures/boxy_sersic_isophotes-light.png#gh-light-mode-only)
![Three images of one Sérsic profile with discy, elliptical and boxy isophotes, with three isophotes drawn on each, and the outer isophote of each drawn together](figures/boxy_sersic_isophotes-dark.png#gh-dark-mode-only)

*One Sérsic profile (n = 2, Rₑ = 10 px, q = 0.7, θ = 30°) with c0 = −0.6, 0 and +0.6, before any PSF. The contours are the isophotes m = 5, 10 and 20 px; the last panel draws the isophote m = 20 px of the three together.*

![Two plots against the distance from the centre: the surface brightness along the diagonal of the isophotes for discy, elliptical and boxy isophotes with the major-axis profile, and their ratio to the elliptical case](figures/boxy_sersic_profiles-light.png#gh-light-mode-only)
![Two plots against the distance from the centre: the surface brightness along the diagonal of the isophotes for discy, elliptical and boxy isophotes with the major-axis profile, and their ratio to the elliptical case](figures/boxy_sersic_profiles-dark.png#gh-dark-mode-only)

*The same profile in units of its central intensity, before any PSF. Along the major axis it is the same for every c0; along the diagonal y′ = q x′ boxy isophotes hold more light and discy ones less, 31 percent more and 41 percent less than ellipses at 30 px. At a fixed flux the central intensity also changes, falling as the isophotes enclose more area.*

## What it represents

The departure of the isophotes from ellipses at fixed axis ratio, measured
on the whole profile. In the Virgo ellipticals of
[Kormendy et al. (2009)][kormendy2009] the
galaxies with depleted cores have boxy isophotes and those with central
extra light have discy ones. The shape is the same at every radius: an
isophote shape that changes with radius, or with a separate disc, needs a
model with two components.

## Known degeneracies

- `c0` and `q` both shape the isophotes and are correlated; the small
  values typical of ellipticals need a high signal-to-noise ratio to be
  measured.
- For `q = 1` and `c0 ≠ 0` the isophotes have fourfold symmetry, so `theta`
  has period π/2 rather than π; for `c0 = 0` and `q = 1` it is undefined.
- The degeneracies of the Sérsic profile between `flux`, `re` and `n` hold
  as for [`single_sersic`](SINGLE_SERSIC.md).

## Validation

- **Rendering.** Tested on 92 cases with `c0` from −1 to 1, `n` from 1 to 6
  and `re` from 1 to 20 px, under PSFs of 2.3 and 3.5 px FWHM: the largest
  pixel error is 7e-4 of the peak ([details](https://github.com/carlocannarozzophd/BayMorph/blob/main/docs/research/MORPHOLOGY_FAMILIES_RENDERING_2026-10-02.md)).
- **Flux.** The rendered flux agrees with the analytic flux within 1e-8, and
  within 3e-6 at `c0` = −1, where the radius has a kink on the axes.
- **Recovery.** On a simulated image, the posterior of `c0` and `n` covers
  the true values.
- **Cost.** About two and a half times the cost of `single_sersic`.
- **Real data.** Not yet used on real images.

## Configuration

```yaml
fit:
  models:
    - boxy_sersic
  priors:
    boxy_sersic:
      flux:
        type: uniform
        lower: 1000.0
        upper: 100000.0
      c0:
        type: uniform
        lower: -0.3
        upper: 0.3
```

Priors not declared take the defaults above; the values here are an
example, to be chosen for each image. A joint fit of several images does not
accept this model.

## Python

`baymorph.models.GeneralizedSersicComponent` is the intrinsic profile;
`baymorph.fitting.make_boxy_sersic_psf_model` builds the PSF-convolved model
a fit evaluates.

## References

- [Athanassoula E. et al., 1990, MNRAS, 245, 130][athanassoula1990]: generalized ellipses.
- [Erwin P., 2015, ApJ, 799, 226][erwin2015]: the parametrization $c = 2 + c_0$.
- [Kormendy J. et al., 2009, ApJS, 182, 216][kormendy2009]: the isophote shapes of Virgo ellipticals.
- [Peng C. Y. et al., 2002, AJ, 124, 266][peng2002]: the area of generalized ellipses.
- [Peng C. Y. et al., 2010, AJ, 139, 2097][peng2010]: the parametrization $c = 2 + c_0$.
- [Sérsic J. L., 1968, Atlas de Galaxias Australes, Observatorio Astronómico, Córdoba][sersic1968]: the Sérsic profile.

[athanassoula1990]: https://ui.adsabs.harvard.edu/abs/1990MNRAS.245..130A/abstract
[erwin2015]: https://ui.adsabs.harvard.edu/abs/2015ApJ...799..226E/abstract
[kormendy2009]: https://ui.adsabs.harvard.edu/abs/2009ApJS..182..216K/abstract
[peng2002]: https://ui.adsabs.harvard.edu/abs/2002AJ....124..266P/abstract
[peng2010]: https://ui.adsabs.harvard.edu/abs/2010AJ....139.2097P/abstract
[sersic1968]: https://ui.adsabs.harvard.edu/abs/1968adga.book.....S/abstract
