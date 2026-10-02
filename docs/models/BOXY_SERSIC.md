# Boxy or Discy Sérsic (`boxy_sersic`)

A [Sérsic profile](COMPONENTS.md#sersic-profile) whose isophotes are
generalized ellipses, boxy or discy instead of elliptical
([generalized Sérsic](COMPONENTS.md#generalized-sersic-profile)). One of the
families of AD-048: available for single-observation fits.

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
of GALFIT and Imfit (Peng et al. 2002). The area within the isophote $m$ is
$K(c)\,q\,m^2$ with $K(c) = 4\,\Gamma(1 + 1/c)^2 / \Gamma(1 + 2/c)$, so
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

![Three images of one Sérsic profile with discy, elliptical and boxy isophotes, with three isophotes drawn on each](figures/boxy_sersic_isophotes-light.png#gh-light-mode-only)
![Three images of one Sérsic profile with discy, elliptical and boxy isophotes, with three isophotes drawn on each](figures/boxy_sersic_isophotes-dark.png#gh-dark-mode-only)

*One Sérsic profile (n = 2, Rₑ = 10 px, q = 0.7, θ = 30°) with c0 = −0.6, 0 and +0.6, before any PSF. The contours are the isophotes m = 5, 10 and 20 px.*

## What it represents

The departure of the isophotes from ellipses at fixed axis ratio, measured
on the whole profile. In the Virgo ellipticals of Kormendy et al. (2009) the
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

- **Rendering.** On the bench of AD-048, with the criteria of AD-046, the
  model meets every criterion on 92 cases with c0 from −1 to 1, n from 1 to
  6 and re from 1 to 20 px, under effective PSFs of 2.34 and 3.5 px FWHM.
  The largest pixel difference is 6.9e-4 of the peak, and the target-like
  chi-square at most 0.12
  ([note](../research/MORPHOLOGY_FAMILIES_RENDERING_2026-10-02.md)).
- **Flux.** A compact profile holds its analytic flux within 1e-8, and
  within 3e-6 at c0 = −1, where the radius has a kink on the axes.
- **Recovery.** On a simulated image with a Gaussian PSF, the posterior of
  `c0` and `n` covers the true values (`tests/test_boxy_sersic_recovery.py`).
- **Cost.** A rendering of 1013 × 1013 px with an effective PSF of 27 px
  takes 103 ms, against 40 ms for `single_sersic`.
- **Real data.** No fit on real data has been made with this model.

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
example, to be chosen for each image. A joint multi-observation fit rejects
this model before writing any output (AD-048).

## Python

`baymorph.models.GeneralizedSersicComponent` is the intrinsic profile;
`baymorph.fitting.make_boxy_sersic_psf_model` builds the PSF-convolved model
a fit evaluates.
