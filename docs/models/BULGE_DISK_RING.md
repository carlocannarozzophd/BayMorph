# Bulge, Disc and Ring (`bulge_disk_ring`)

A Sérsic bulge, an exponential disc and an elliptical ring with a Gaussian
radial profile ([Gaussian ring](COMPONENTS.md#gaussian-ring)), sharing a
centre. One of the families of AD-048: available for single-observation
fits.

## Definition

$$
I(\mathbf{x}) = I_{\mathrm{Sersic}}(\mathbf{x};\,F_b, R_{e,b}, n_b, q_b,
\theta_b) + I_{\mathrm{exp}}(\mathbf{x};\,F_d, R_{e,d}, q_d, \theta_d)
+ I_{\mathrm{ring}}(\mathbf{x};\,F_r, r, \sigma, q_r, \theta_r),
$$

the bulge and the disc of [`bulge_disk`](BULGE_DISK.md) and the ring of
Imfit ([Erwin 2015][erwin2015]),

$$
I_{\mathrm{ring}}(m) = A \exp\left[-\frac{(m - r)^{2}}{2\sigma^{2}}\right],
$$

on ellipses of elliptical radius $m$ and axis ratio $q_r$. The crest lies at
the semi-major radius $r$ and $\sigma$ is the width along the major axis. The
flux of the ring is

$$
F_r = 2\pi q_r A \left\{\sigma^{2} e^{-r^{2}/(2\sigma^{2})}
+ \sqrt{\frac{\pi}{2}}\, r\sigma
\left[1 + \operatorname{erf}\left(\frac{r}{\sqrt{2}\,\sigma}\right)
\right]\right\}.
$$

The ring has its own axis ratio and position angle, so it may be
intrinsically elliptical or lie in another plane than the disc.

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
| `flux_ring` | total flux of the ring | image units | none: must be declared |
| `r_ring` | semi-major radius of the crest of the ring | pixels | uniform 5 to 200 |
| `sigma_ring` | width of the ring along its major axis | pixels | uniform 2 to 20 |
| `q_ring` | axis ratio of the ring | | uniform 0.5 to 1 |
| `theta_ring` | position angle of the ring | radians | uniform −π/2 to π/2 |
| `x0`, `y0` | shared centre | pixels | uniform ±8 about the target seed |

The default priors are generic and in pixels; they must be checked against
the target and the image. `sigma_ring` must be at least 2 px and `q_ring` at
least 0.5: the domain in which the rendering of the ring is validated, which
configuration validation enforces.

![A plot of three rings of the same flux and different widths against radius, and the image of an inclined ring](figures/gaussian_ring_profiles-light.png#gh-light-mode-only)
![A plot of three rings of the same flux and different widths against radius, and the image of an inclined ring](figures/gaussian_ring_profiles-dark.png#gh-dark-mode-only)

*Left: rings at r = 20 px with the same flux and σ of 1, 3 and 6 px, along the major axis; a wider ring has a lower crest. Right: a ring with r = 26 px, σ = 2.5 px and q = 0.6.*

![Images of the convolved bulge, disc, ring and their sum, with the radial profiles of the four](figures/bulge_disk_ring_decomposition-light.png#gh-light-mode-only)
![Images of the convolved bulge, disc, ring and their sum, with the radial profiles of the four](figures/bulge_disk_ring_decomposition-dark.png#gh-dark-mode-only)

*A bulge of 25 percent of the light (n = 3, Rₑ = 3 px), a disc of 60 percent (Rₑ = 16 px) and a ring of 15 percent (r = 26 px, σ = 2.5 px, q = 0.8), after a Gaussian PSF of 2.5 px FWHM. The profiles are averages over circles.*

## What it represents

A disc galaxy with a bulge and an inner or outer ring, or the ring-like
residual that a bulge and disc decomposition leaves when the galaxy has
one. Imfit (Erwin 2015) provides the same Gaussian ring for such
decompositions.

## Known degeneracies

- A wide ring adds light over a range of radii of the disc, so `sigma_ring`,
  `flux_ring` and `re_disk` are correlated; a ring needs to be narrower than
  the disc to be told apart from it.
- At low signal-to-noise, `flux_ring` and `sigma_ring` are correlated: a
  faint wide ring and a fainter narrow one differ little in each pixel.
- `q_ring` and `theta_ring` are free. For an intrinsically circular ring in
  the plane of the disc they equal `q_disk` and `theta_disk`, which the
  posterior can show but the model does not impose.

## Validation

- **Rendering.** On the bench of AD-048, with the criteria of AD-046, 70
  cases under effective PSFs of 2.34 and 3.5 px FWHM. Rings of σ = 1 px
  fail criterion 1 far from the centre, at r = 25 px, when q is 0.5 or
  less: they are half a pixel wide or less across their crest on the minor
  axis, and outside the window of AD-046 the profile is sampled at pixel
  centres. Every ring of σ = 2 or 5 px meets every criterion of the grid,
  with a largest difference of 1.5e-4 of the peak against a limit of 1e-2
  ([note](../research/MORPHOLOGY_FAMILIES_RENDERING_2026-10-02.md)).
- **Target-like cases.** At the signal-to-noise of the r-band target the
  ring alone reaches a chi-square of 0.03 to 0.04. With the compact bulge of
  these cases, 30 percent of the flux, the total reaches 1.04 at 2.34 px
  FWHM: the limit of the production Sérsic profile found with
  [`bulge_broken_disk`](BULGE_BROKEN_DISK.md).
- **Flux.** A ring inside the image holds its analytic flux within 1e-8,
  and its pixel integrals agree with adaptive integration within 6e-8 of
  the peak.
- **Recovery.** On a simulated galaxy with a ring, the posterior of `r_ring`
  and `sigma_ring` covers the true values
  (`tests/test_bulge_disk_ring_recovery.py`).
- **Cost.** A rendering of 1013 × 1013 px with an effective PSF of 27 px
  takes 67 ms, against 54 ms for `bulge_disk`.
- **Real data.** No fit on real data has been made with this model.

## Configuration

```yaml
fit:
  models:
    - bulge_disk_ring
  priors:
    bulge_disk_ring:
      flux_bulge:
        type: uniform
        lower: 1000.0
        upper: 100000.0
      flux_disk:
        type: uniform
        lower: 1000.0
        upper: 200000.0
      flux_ring:
        type: uniform
        lower: 500.0
        upper: 50000.0
```

Priors not declared take the defaults above; the values here are an
example, to be chosen for each image. A joint multi-observation fit rejects
this model before writing any output (AD-048).

## Python

`baymorph.models.GaussianRingComponent` is the ring;
`baymorph.fitting.make_bulge_disk_ring_psf_model` builds the PSF-convolved
model a fit evaluates.

## References

- [Erwin P., 2015, ApJ, 799, 226][erwin2015]: the Gaussian ring.
- [Freeman K. C., 1970, ApJ, 160, 811][freeman1970]: the exponential disc.
- [Sérsic J. L., 1968, Atlas de Galaxias Australes, Observatorio Astronómico, Córdoba][sersic1968]: the Sérsic profile of the bulge.

[erwin2015]: https://ui.adsabs.harvard.edu/abs/2015ApJ...799..226E/abstract
[freeman1970]: https://ui.adsabs.harvard.edu/abs/1970ApJ...160..811F/abstract
[sersic1968]: https://ui.adsabs.harvard.edu/abs/1968adga.book.....S/abstract
