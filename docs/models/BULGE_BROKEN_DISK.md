# Bulge and Broken Disc (`bulge_broken_disk`)

A Sérsic bulge and an exponential disc whose scale length changes at a break
radius ([broken exponential disc](COMPONENTS.md#broken-exponential-disc)),
sharing a centre. Available for fits of a single image.

## Definition

$$
I(\mathbf{x}) = I_{\mathrm{Sersic}}(\mathbf{x};\,F_b, R_{e,b}, n_b, q_b,
\theta_b) + I_{\mathrm{disc}}(\mathbf{x};\,F_d, h_{\mathrm{in}},
h_{\mathrm{out}}, r_b, w_b, q_d, \theta_d),
$$

with the disc of [Erwin, Pohlen & Beckman (2008)][erwin2008],

$$
I_{\mathrm{disc}}(r) = I_0 \exp\left(-\frac{r}{h_{\mathrm{in}}}\right)
\left[\frac{1 + e^{(r - r_b)/w_b}}{1 + e^{-r_b/w_b}}\right]^{
w_b (1/h_{\mathrm{in}} - 1/h_{\mathrm{out}})},
$$

an exponential of scale length $h_{\mathrm{in}}$ inside the break radius
$r_b$ and of $h_{\mathrm{out}}$ outside it, joined over a width $w_b$. Erwin,
Pohlen & Beckman write the sharpness $\alpha = 1/w_b$; BayMorph uses the
width, so that every disc parameter but the flux is a length. The disc is a
down-bending type II for $h_{\mathrm{out}} < h_{\mathrm{in}}$
([Freeman 1970][freeman1970]), an up-bending type III for
$h_{\mathrm{out}} > h_{\mathrm{in}}$
([Erwin, Beckman & Pohlen 2005][erwin2005]), and a single exponential,
type I, for $h_{\mathrm{out}} = h_{\mathrm{in}}$.

## Parameters

| Parameter | Meaning | Units | Default prior |
| --- | --- | --- | --- |
| `flux_bulge` | total flux of the bulge | image units | none: must be declared |
| `re_bulge` | half-light radius of the bulge | pixels | uniform 1 to 30 |
| `n_bulge` | Sérsic index of the bulge | | uniform 1.5 to 6 |
| `q_bulge` | axis ratio of the bulge | | uniform 0.4 to 1 |
| `theta_bulge` | position angle of the bulge | radians | uniform −π/2 to π/2 |
| `flux_disk` | total flux of the disc | image units | none: must be declared |
| `h_in_disk` | scale length inside the break | pixels | uniform 2 to 80 |
| `h_out_disk` | scale length outside the break | pixels | uniform 2 to 80 |
| `rb_disk` | break radius | pixels | uniform 5 to 300 |
| `wb_disk` | width of the break | pixels | uniform 0.5 to 20 |
| `q_disk` | axis ratio of the disc | | uniform 0.4 to 1 |
| `theta_disk` | position angle of the disc | radians | uniform −π/2 to π/2 |
| `x0`, `y0` | shared centre | pixels | uniform ±8 about the target seed |

The default priors are generic and in pixels; they must be checked against
the target and the image. `wb_disk` must be at least 0.5 px, the sharpest
break whose rendering is validated, which configuration validation enforces.

![Two plots of the logarithm of the disc profile against radius: types I, II and III with the same inner scale length, and a type II disc with three widths of the break](figures/broken_disk_profiles-light.png#gh-light-mode-only)
![Two plots of the logarithm of the disc profile against radius: types I, II and III with the same inner scale length, and a type II disc with three widths of the break](figures/broken_disk_profiles-dark.png#gh-dark-mode-only)

*Left: discs of types I, II and III with $h_{\mathrm{in}}$ = 6 px and a break at 15 px. Right: a type II disc with three widths of the break.*

![Images of the convolved bulge, the type II disc and their sum, with the radial profiles of the three](figures/bulge_broken_disk_decomposition-light.png#gh-light-mode-only)
![Images of the convolved bulge, the type II disc and their sum, with the radial profiles of the three](figures/bulge_broken_disk_decomposition-dark.png#gh-dark-mode-only)

*A bulge of 30 percent of the light (n = 3.5, Rₑ = 3 px) and a type II disc ($h_{\mathrm{in}}$ = 10 px, $h_{\mathrm{out}}$ = 4 px, break at 22 px), after a Gaussian PSF of 2.5 px FWHM. The profiles are averages over circles, so the break of the inclined disc appears smoothed.*

## What it represents

A bulge and a disc whose outer profile departs from a single exponential.
Among the late-type spirals of
[Pohlen & Trujillo (2006)][pohlen-trujillo2006], about 10 percent
have type I profiles, 60 percent type II and 30 percent type III, and the
fraction of up-bending profiles rises toward earlier types. Up-bending
profiles are also found in early-type barred discs (Erwin, Beckman & Pohlen
2005). A break needs depth to be measured: it lies at a few scale lengths,
where the disc is faint.

## Known degeneracies

- The bulge and the inner disc share the light at small radii, as in
  [`bulge_disk`](BULGE_DISK.md).
- `h_out_disk` and the residual background are anticorrelated, because both
  set the light far from the centre.
- `rb_disk` and `wb_disk` are correlated for broad breaks, and a break within
  the bulge-dominated radii is weakly constrained.
- For `h_out_disk` near `h_in_disk` the break vanishes, and `rb_disk` and
  `wb_disk` follow their priors.

## Validation

- **Rendering.** The disc was tested on 96 cases of types II and III, with
  `wb_disk` from 0.5 to 5 px, under PSFs of 2.3 and 3.5 px FWHM: the largest
  pixel error is 3e-5 of the peak.
- **Compact bright bulges.** The bulge is rendered as in
  [`bulge_disk`](BULGE_DISK.md): a compact bright bulge is rendered with an
  error comparable with the noise above a peak signal-to-noise of about
  1.5e4.
- **Flux.** A compact disc holds its analytic flux within 1e-8. With
  `h_out_disk` equal to `h_in_disk` the disc reproduces the exponential disc
  within 2.3e-7 of the peak.
- **Recovery.** On a simulated type II disc with a bulge, the posterior of
  `h_out_disk` and `rb_disk` covers the true values.
- **Cost.** About a quarter more than `bulge_disk`.
- **Real data.** Not yet used on real images.

## Configuration

```yaml
fit:
  models:
    - bulge_broken_disk
  priors:
    bulge_broken_disk:
      flux_bulge:
        type: uniform
        lower: 1000.0
        upper: 100000.0
      flux_disk:
        type: uniform
        lower: 1000.0
        upper: 200000.0
```

Priors not declared take the defaults above; the values here are an
example, to be chosen for each image. A joint fit of several images does not
accept this model.

## Python

`baymorph.models.BrokenExponentialComponent` is the disc;
`baymorph.fitting.make_bulge_broken_disk_psf_model` builds the PSF-convolved
model a fit evaluates.

## References

- [Erwin P., Beckman J. E., Pohlen M., 2005, ApJ, 626, L81][erwin2005]: the type III profile.
- [Erwin P., Pohlen M., Beckman J. E., 2008, AJ, 135, 20][erwin2008]: the broken exponential disc.
- [Freeman K. C., 1970, ApJ, 160, 811][freeman1970]: the exponential disc and the type II profile.
- [Pohlen M., Trujillo I., 2006, A&A, 454, 759][pohlen-trujillo2006]: the frequency of the three types.
- [Sérsic J. L., 1968, Atlas de Galaxias Australes, Observatorio Astronómico, Córdoba][sersic1968]: the Sérsic profile of the bulge.

[erwin2005]: https://ui.adsabs.harvard.edu/abs/2005ApJ...626L..81E/abstract
[erwin2008]: https://ui.adsabs.harvard.edu/abs/2008AJ....135...20E/abstract
[freeman1970]: https://ui.adsabs.harvard.edu/abs/1970ApJ...160..811F/abstract
[pohlen-trujillo2006]: https://ui.adsabs.harvard.edu/abs/2006A%26A...454..759P/abstract
[sersic1968]: https://ui.adsabs.harvard.edu/abs/1968adga.book.....S/abstract
