# Point-Spread Functions

The point-spread function belongs to the observation, not to the galaxy: it
maps the intrinsic light to the blurred light the detector records. Each PSF
declares whether its kernel already contains the response of a detector
pixel, and that declaration decides how a model is rendered with it.
`psf.type` selects one of two PSFs.

## Gaussian PSF (`gaussian`)

**Definition.** A circular Gaussian,

$$
P(r) \propto \exp\left(-\frac{r^2}{2\sigma^2}\right),
\qquad
\sigma = \frac{\mathrm{FWHM}}{2\sqrt{2\ln 2}} = \frac{\mathrm{FWHM}}{2.3548},
$$

evaluated at integer offsets on a kernel of 25 x 25 px and normalized to
unit sum.

| Key | Meaning | Units |
| --- | --- | --- |
| `psf.gaussian.fwhm` | full width at half maximum | pixels |

![The 25 by 25 pixel Gaussian kernel as an image, and its pixel values against radius with the continuous Gaussian and the half maximum](figures/gaussian_psf-light.png#gh-light-mode-only)
![The 25 by 25 pixel Gaussian kernel as an image, and its pixel values against radius with the continuous Gaussian and the half maximum](figures/gaussian_psf-dark.png#gh-dark-mode-only)

*A Gaussian PSF of 3 px FWHM: the kernel and its pixel values, normalized to the peak, against the continuous Gaussian.*

**Rendering.** The kernel is an optical PSF: it does not contain the pixel
response, so a profile is integrated over each pixel before convolution, and
a point source is placed by shifting the kernel. The profile is rendered on
a domain extended by the half-width of the kernel, so that light from
outside the cutout scattered into it is kept.

**Use and limits.** It suits simulations and images whose PSF is known to be
Gaussian. It is circular and constant across the image, and a Gaussian much
wider than 8 px FWHM loses part of its wings to the 25 px kernel. It is not
a substitute for an empirical PSF of a real image.

**Python.** `baymorph.GaussianPSF(fwhm)`.

## Empirical PSF (`empirical`)

**Definition.** A kernel built from the stars of the image. Because it is
made of detector pixels, it samples the *effective* PSF, the optical PSF
convolved with the pixel response, as defined by
[Anderson & King (2000)][anderson-king2000]; the builder below is that of
BayMorph, not theirs. The product holds
the native kernel and the effective PSF at the eleven sub-pixel phases of
each axis; every phase is non-negative with unit sum, and the zero phase
equals the native kernel.

**Construction.** `select_psf_candidates()` selects isolated, round stars
whose area and peak lie within declared limits
(`psf.empirical.candidate_selection`); `build_psf()` then:

1. fits each kernel pixel as a weighted quadratic in the sub-pixel phase of
   the stars, with no interpolation;
2. measures the sky of each star in a ring between `sky_inner_radius` and
   `sky_outer_radius`, 80 and 160 px by default, with the other detected
   sources masked after growing them by `mask_growth_pixels`, 5 px;
3. continues the kernel beyond the stars' common footprint with their
   measured azimuthal profile, out to `support_half_size`, 64 px, so that
   the default kernel is 129 x 129 px.

**Rendering.** Since the kernel already contains the pixel response, a model
follows the effective-PSF rendering. A smooth window, one minus a
cumulative normal centred at 7 px with a width of 1 px and set to zero
beyond 14 px, splits the profile about each component centre. Within it the
profile is integrated over cells eleven times finer and convolved with the
sub-pixel kernels; outside it, it is sampled at pixel centres and convolved
with the native kernel. A point source is placed by interpolating between
the sub-pixel kernels. An empirical PSF without sub-pixel kernels, such as
one built by an earlier version of BayMorph, is rejected: build it again.

![Left: the window weight w and its complement 1 minus w against the distance from the component centre, crossing at 7 pixels and cut at 14 pixels. Right: the window as a map, with circles at 7 and 14 pixels](figures/effective_psf_window-light.png#gh-light-mode-only)
![Left: the window weight w and its complement 1 minus w against the distance from the component centre, crossing at 7 pixels and cut at 14 pixels. Right: the window as a map, with circles at 7 and 14 pixels](figures/effective_psf_window-dark.png#gh-dark-mode-only)

*The split of the effective-PSF rendering: the weight w goes to cells eleven times finer convolved with the sub-pixel kernels, and 1 − w to samples at pixel centres convolved with the native kernel. The two halves cross at 7 px and the window is cut at 14 px.*

**Validation.** On synthetic star fields of known PSF, the builder keeps the
bias of `n` and `re` within 1 percent, and the rendering recovers every
parameter of simulated galaxies within 0.09 standard deviations.
The validated domain is a circular PSF constant across the field, with a
Moffat core ([Moffat 1969][moffat1969]) of 2.2 to 3.5 px FWHM with or
without a halo, stars at least 50 px apart, no light beyond 64 px, and
profiles with an effective radius of
at least 1 px and an index up to 6. The extended wings leave a
field-to-field uncertainty of about 1 percent in `re`, which a fit with one
kernel cannot average away and must state as a systematic uncertainty.

**Limits.** One kernel describes the whole image; a PSF that varies across
the field is not modelled. Real PSFs may extend further than 64 px, and real
fields are more crowded than the validated ones.

**Python.** `baymorph.EmpiricalPSF(kernel_array, subpixel_kernels=...)`.

## References

- [Anderson J., King I. R., 2000, PASP, 112, 1360][anderson-king2000]: the effective PSF.
- [Moffat A. F. J., 1969, A&A, 3, 455][moffat1969]: the Moffat profile of the validated domain.

[anderson-king2000]: https://ui.adsabs.harvard.edu/abs/2000PASP..112.1360A/abstract
[moffat1969]: https://ui.adsabs.harvard.edu/abs/1969A%26A.....3..455M/abstract
