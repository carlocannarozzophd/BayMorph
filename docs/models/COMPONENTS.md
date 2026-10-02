# Morphological Components

The morphology models are sums of three components: the Sérsic profile, the
exponential disc and the unresolved point source. A component describes
intrinsic light only; it owns no PSF, no noise and no prior.

## Sérsic profile

**Definition.** About its centre $(x_0, y_0)$, the elliptical radius is

$$
R^2 = x'^2 + \left(\frac{y'}{q}\right)^2,
\qquad
\begin{aligned}
x' &= (x - x_0)\cos\theta + (y - y_0)\sin\theta,\\
y' &= -(x - x_0)\sin\theta + (y - y_0)\cos\theta,
\end{aligned}
$$

and the surface brightness is

$$
I(R) = I_e \exp\left\{-b_n\left[\left(\frac{R}{R_e}\right)^{1/n} - 1\right]\right\}.
$$

$b_n$ makes $R_e$ the half-light radius: it solves
$\gamma(2n, b_n) = \Gamma(2n)/2$ and is computed exactly as the inverse of
the regularized incomplete gamma function, not from an approximation. It is
0.693 for $n = 0.5$, 1.678 for $n = 1$, 3.672 for $n = 2$ and 7.669 for
$n = 4$.

![Diagram of an elliptical isophote on the pixel grid: centre (x0, y0), half-light radius along the major axis, minor semi-axis q times Re, and the position angle theta measured from +x towards +y](figures/geometry-light.png#gh-light-mode-only)
![Diagram of an elliptical isophote on the pixel grid: centre (x0, y0), half-light radius along the major axis, minor semi-axis q times Re, and the position angle theta measured from +x towards +y](figures/geometry-dark.png#gh-dark-mode-only)

*The conventions of every component: pixel [j, i] is centred at (x, y) = (i, j); Rₑ lies along the major axis, the minor semi-axis is q Rₑ, and θ runs from +x towards +y, with +y up as BayMorph displays images.*

![Two plots: surface brightness relative to its value at Re against R/Re for n from 0.5 to 6, and the fraction of light enclosed within R](figures/sersic_profiles-light.png#gh-light-mode-only)
![Two plots: surface brightness relative to its value at Re against R/Re for n from 0.5 to 6, and the fraction of light enclosed within R](figures/sersic_profiles-dark.png#gh-dark-mode-only)

*Left: the profile for indices from 0.5 to 6; a higher index means a steeper core and more extended wings. Right: the light enclosed within R, which is one half at R = Rₑ for every index.*

**Normalization.** The amplitude follows from the total flux of the
continuous profile over the full plane,

$$
F = 2\pi q I_e R_e^2\, n\, e^{b_n} b_n^{-2n}\, \Gamma(2n),
$$

so `flux` is a property of the galaxy, independent of the cutout, the pixel
scale, the PSF and the backend.

**Parameters.**

| Parameter | Meaning | Units | Domain |
| --- | --- | --- | --- |
| `flux` | total flux of the continuous profile | image units | $F > 0$ |
| `re` | half-light radius along the major axis | pixels | $R_e > 0$ |
| `n` | Sérsic index | | $n > 0$ |
| `q` | axis ratio $b/a$ | | $0 < q \le 1$ |
| `theta` | position angle of the major axis, from $+x$ towards $+y$ | radians | finite, period π |
| `x0`, `y0` | centre | pixels of the cutout | finite |

**Rendering.** Detector pixels are finite regions, so the expected value of
a pixel is an integral of the profile, and the PSF decides which integral:

- with a PSF that does not include the pixel response, such as the
  Gaussian, the profile is integrated over each pixel before convolution.
  The production integrator, `hybrid`, uses Gauss-Legendre quadrature away
  from the centre and semi-analytic pixel integrals about the cusp; it was
  measured accurate to 6e-9 of the total flux;
- with an empirical PSF, which already contains the pixel response, the
  profile follows the effective-PSF rendering of AD-046: within a smooth
  window about the centre it is integrated over cells eleven times finer and
  convolved with the sub-pixel kernels of the PSF, and outside it it is
  sampled at pixel centres and convolved with the native kernel. See
  [Point-Spread Functions](PSF.md) and
  [Rendering](../architecture/RENDERING.md), §13.1.

**Python.** `baymorph.SersicModel(flux, re, n, q, theta, x0, y0)`; its
`render(shape)` integrates the profile over the pixels of a grid.

## Exponential disc

**Definition.** The Sérsic profile with $n = 1$ fixed:

$$
I(R) = I_e \exp\left[-b_1\left(\frac{R}{R_e} - 1\right)\right],
\qquad b_1 = 1.678.
$$

Its `re` is the half-light radius, not the exponential scale length $h$:
$h = R_e / b_1 \approx 0.596\, R_e$. Its parameters are those of the Sérsic
profile without `n`.

**Python.** `baymorph.models.ExponentialComponent(flux, re, q, theta, x0, y0)`.

## Point source

**Definition.** An unresolved source is a Dirac distribution,

$$
I(\mathbf{x}) = F\,\delta(\mathbf{x} - \mathbf{x}_0),
$$

so its expected image is the PSF itself, scaled by the flux and placed at
the source position. It is never approximated by a narrow Sérsic or
Gaussian profile, and it is never convolved twice.

| Parameter | Meaning | Units | Domain |
| --- | --- | --- | --- |
| `flux` | total flux before the PSF | image units | $F > 0$ |
| `x0`, `y0` | position | pixels of the cutout | finite |

**Rendering.** With an empirical PSF the source is placed by interpolating
between the sub-pixel kernels of the PSF (AD-046). With a Gaussian PSF the
normalized kernel is shifted by cubic-spline interpolation; the negative
ringing this introduces is clipped only while it stays below 1 percent of
the flux, and an error is raised beyond. Flux that falls outside the image
is lost, not redistributed.

![Three small images of the same point source rendered through a Gaussian PSF, with its centre 0, 0.25 and 0.5 pixel from a pixel centre](figures/point_source_placement-light.png#gh-light-mode-only)
![Three small images of the same point source rendered through a Gaussian PSF, with its centre 0, 0.25 and 0.5 pixel from a pixel centre](figures/point_source_placement-dark.png#gh-dark-mode-only)

*A point source through a Gaussian PSF of 2.5 px FWHM, offset from a pixel centre by 0, 0.25 and 0.5 px: the light is shared among pixels according to the true position, marked by the dot.*

**Python.** `baymorph.models.PointSourceModel(flux, x0, y0)`; it has no
`render()` method, because it needs a PSF to be rendered.
