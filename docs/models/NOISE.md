# Noise Models

The noise model is the probability of the observed image given the expected
image of a model: the likelihood of the fit. It decides how much each pixel
counts and how wide the posterior is, and its normalization makes the
Bayesian evidence absolute. `fit.likelihood.type` selects it; every fit run
from a configuration uses the correlated Gaussian below.

## Correlated Gaussian (`correlated_gaussian`)

**Definition.** Over the $N$ pixels the mask leaves in, with residuals
$\mathbf{r} = \mathbf{D} - \boldsymbol{\mu}$ between the data and the
expected image,

$$
\ln L = -\frac{1}{2}\,\mathbf{r}^{\mathsf{T}} C^{-1}\,\mathbf{r}
- \frac{1}{2}\ln\det C - \frac{N}{2}\ln 2\pi,
\qquad
C = S\,R\,S,
$$

a Gaussian whose covariance has a diagonal scale
$S = \mathrm{diag}(\sigma_i)$ and a correlation $R$ with unit diagonal. The
scale of pixel $i$ adds the noise of the sky and the Poisson noise of the
galaxy,

$$
\sigma_i^2 = s^2 + \frac{\max(m_i, 0)}{g},
$$

the variance equation of a CCD image after sky subtraction
([Newberry 1991][newberry1991]), where $m_i$ is the model of the galaxy in
that pixel, the expected image without the residual background.

| Input | Meaning | Where it comes from |
| --- | --- | --- |
| $s$ | the noise scale of the sky | measured by `prepare_image()` on the sky region |
| $g$ | the effective gain: detected photons, or electrons, per image unit | declared in `fit.likelihood.effective_gain`; see [Sky and Noise](../guide/SKY_AND_NOISE.md) |
| $R$ | the correlation of the sky between pixels | measured on the sky region before the first fit |

None of them is sampled: they are fixed before the fit, and the noise model
adds no parameter to it.

![Two panels: the noise of one pixel against the model of the galaxy for three effective gains, flat at the sky noise and rising as the square root of the model, and the map of the noise of each pixel of a Sérsic galaxy](figures/noise_variance-light.png#gh-light-mode-only)
![Two panels: the noise of one pixel against the model of the galaxy for three effective gains, flat at the sky noise and rising as the square root of the model, and the map of the noise of each pixel of a Sérsic galaxy](figures/noise_variance-dark.png#gh-dark-mode-only)

*Left: the noise of one pixel for a sky noise of 20 image units: the sky sets a floor, and the Poisson noise of the galaxy takes over where its light exceeds about $g\,s^2$. Right: the noise of each pixel of a bright Sérsic galaxy with g = 1, highest at its centre.*

**The correlation of the sky.** Resampling and coadding images spread the
noise of one pixel over its neighbours
([Casertano et al. 2000][casertano2000];
[Fruchter & Hook 2002][fruchter-hook2002]). BayMorph measures that
correlation once, on the sky region of the prepared cutout: it removes a
plane from the sky pixels, computes their autocovariance at every lag up to
8 pixels along each axis, normalized by the number of pixel pairs and
without a taper, and divides it by its value at zero lag. The correlation is
assumed the same everywhere in the cutout. A sky region with fewer than 1000
pixel pairs at some lag up to 16, or a correlation that is not a valid
covariance, stops the fit instead of being repaired.

![Four panels: a patch of white noise, a patch of correlated noise, the correlation of both against the lag along x, falling to zero at lag 1 for white noise and over three pixels for correlated noise, and a grid showing a pixel and the 24 earlier pixels it is conditioned on](figures/correlated_noise-light.png#gh-light-mode-only)
![Four panels: a patch of white noise, a patch of correlated noise, the correlation of both against the lag along x, falling to zero at lag 1 for white noise and over three pixels for correlated noise, and a grid showing a pixel and the 24 earlier pixels it is conditioned on](figures/correlated_noise-dark.png#gh-dark-mode-only)

*Top: white noise and noise correlated over about a pixel, as in a resampled image. Bottom left: their correlation along x, measured by the estimator of BayMorph. Bottom right: the 24 earlier pixels on which the factorization conditions a pixel.*

**A sky with structure on large scales.** The estimator above assumes that
the correlation ends within 8 pixels. Many real skies also hold structure on
scales of tens to hundreds of pixels, left by the subtraction of the sky
during the reduction or by the faint outskirts of sources: their
correlation is still near 0.05 to 0.1 at 8 pixels, the kernel cut there is
not a valid covariance, and the fit stops. For such a sky BayMorph offers a
second estimator, which you declare; it is never chosen automatically:

```yaml
preparation:
  noise:
    sky_correlation: field_separated
```

It measures the correlation of the sky up to 48 pixels, fits a smooth
correlation $v\,e^{-r/\ell} + c$ to its values beyond 8 pixels, with $\ell$
between 16 and 400 pixels, subtracts it within the kernel and divides the
rest by its value at zero lag, $1 - v - c$. The constant $c$ absorbs the
deficit that the removal of the plane leaves at short lags. The structure
it removes is not modelled by the likelihood, which then describes the noise
of the sky alone. The estimator was validated on sky regions whose masks
were grown by 20 pixels, the default growth ([Masks](../guide/MASKS.md)),
and it requires that growth, so it cannot be used with a curated mask in
`exact` mode.

**The nearest-neighbour factorization.** The inverse of $C$ is out of reach
for an image of a million pixels, so BayMorph replaces the joint Gaussian by
a product of conditional Gaussians, the approximation of Vecchia
([Katzfuss & Guinness 2021][katzfuss-guinness2021]):

$$
p(\mathbf{r}) \approx \prod_{i=1}^{N}
p\!\left(r_i \,\middle|\, r_j,\ j \in \mathcal{N}(i)\right),
$$

where $\mathcal{N}(i)$ are the 24 pixels nearest to $i$ among those that
come before it, row by row. BayMorph factorizes the correlation $R$ once,
before the fit, and stores the result with the preparation; at every
evaluation it divides the residual of each pixel by its $\sigma_i$ and
applies that factorization. Each factor is a normalized Gaussian, so the
likelihood stays normalized and the evidence keeps its absolute meaning, up
to the approximation. On an image of about a million pixels the noise model
adds about 10 ms to each evaluation of the likelihood.

**What it represents.** The noise of the sky, the Poisson noise of the
galaxy and the correlation of neighbouring sky pixels. In a joint fit of
several images, each image has its own noise model, and the images are
assumed independent of each other.

**Limits.**

- $s$, $g$ and $R$ are fixed inputs: their uncertainty is not propagated
  into the posterior.
- An external weight or variance map is not yet supported, so a coadd of
  variable depth is described by one sky noise.
- The correlation is measured on the sky region of the cutout: a small
  cutout gives a noisy estimate, and faint light of the galaxy beyond the
  plateau, if any, raises it.
- With `field_separated`, the line between the noise and the large-scale
  structure is an assumption: structure between about 8 and 30 pixels can
  be counted as either. Each calibration records as `attribution_ratio` the
  kernel sum of a high-pass estimate, which counts that structure as noise,
  over the kernel sum it uses; on three real coadds the ratio was 1.17, 1.29
  and 2.23. The large-scale structure also adds an error on the scale of the
  galaxy that no term of the likelihood carries, so on such a sky the
  posterior intervals are too narrow.
- The Poisson term uses the model, not the data, so it never depends on the
  noise of the image itself.

**Validation.** On simulated single exposures and coadds, resampled, with
correlated sky structure or with variable depth, a model of independent
pixels with the sky noise alone reported intervals of the flux, the radius
and the index 1.1 to 3.0 times too narrow. This noise model brought the
ratio of the true to the reported width to between 0.90 and 1.16, and its
posterior widths agreed with the curvature of the likelihood within 5
percent. On the same simulations the gain measured from a sky still in the
image was within 0.6 percent of the truth.
On simulated skies with a large-scale field of 7 percent of the sky
variance, `field_separated` recovered the correlation of the noise within 8
percent on average, and within 3 percent without a field, where the default
estimator applies. Fitted with masks grown by 20 pixels, the flux, the
effective radius and the Sérsic index were biased by less than 3 percent,
but their reported intervals were 6 to 8 times too narrow.
No fit of a real image has been validated yet.

**Configuration.**

```yaml
fit:
  likelihood:
    type: correlated_gaussian
    effective_gain:
      source: measured_sky     # or declared, sky_level, absent
```

The factorization, `nearest_neighbor` with 24 neighbours, and the complete
normalization are the defaults and the only values a fit accepts.

## Other likelihoods

| Type | Describes | Status |
| --- | --- | --- |
| `gaussian` | independent pixels with the sky noise alone | refused for every fit run from a configuration, because its intervals are too narrow on real images; available through the Python API for simulations and tests |
| `cash` | the Poisson likelihood of an image in counts ([Cash 1979][cash1979]) | experimental: written for images in counts, not validated on the prepared images of a fit |
| `poisson_gaussian` | Poisson counts with Gaussian read noise, marginalized over the counts | experimental, as `cash` |

Every fit requires a completely normalized likelihood, so the evidence is
always absolute: `cash` must keep its data term.

## References

- [Casertano S. et al., 2000, AJ, 120, 2747][casertano2000]: the correlated noise of resampled images.
- [Cash W., 1979, ApJ, 228, 939][cash1979]: the Poisson likelihood.
- [Fruchter A. S., Hook R. N., 2002, PASP, 114, 144][fruchter-hook2002]: the resampling of dithered images.
- [Katzfuss M., Guinness J., 2021, Statistical Science, 36, 124][katzfuss-guinness2021]: the approximation of Vecchia.
- [Newberry M. V., 1991, PASP, 103, 122][newberry1991]: the noise of a sky-subtracted CCD image.

[casertano2000]: https://ui.adsabs.harvard.edu/abs/2000AJ....120.2747C/abstract
[cash1979]: https://ui.adsabs.harvard.edu/abs/1979ApJ...228..939C/abstract
[fruchter-hook2002]: https://ui.adsabs.harvard.edu/abs/2002PASP..114..144F/abstract
[katzfuss-guinness2021]: https://ui.adsabs.harvard.edu/abs/2017arXiv170806302K/abstract
[newberry1991]: https://ui.adsabs.harvard.edu/abs/1991PASP..103..122N/abstract
