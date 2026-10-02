# Morphological Components

The four original morphology models are sums of three components: the
Sérsic profile, the exponential disc and the unresolved point source. The
families of AD-048 add components of their own, described at the end of
this page. A component describes intrinsic light only; it owns no PSF, no
noise and no prior.

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

## Components of AD-048

The families of AD-048 render through generic cell rules
(`baymorph/rendering/profile_integration.py`) with the renderer names of the
Sérsic profile, so every PSF renders them as it renders the Sérsic profile:

- under an analytic PSF, Gauss-Legendre quadrature over the pixels;
- under an empirical PSF, the effective-PSF rendering of AD-046;
- in both cases, adaptive integration about the centre and on the cells a
  component declares non-smooth;
- for a component whose light ends at an edge, cells the edge crosses split
  until they are small, since the test of agreement of adaptive integration
  cannot see a sliver of light that no node samples.

Their `flux` is the total flux over the whole plane, as for the Sérsic
profile. Each is validated within a rendering domain recorded in
[Rendering of the Morphology Families](../research/MORPHOLOGY_FAMILIES_RENDERING_2026-10-02.md).

### Generalized Sérsic profile

**Definition.** The Sérsic profile of the generalized radius

$$
m = \left(|x'|^{c} + \left|\frac{y'}{q}\right|^{c}\right)^{1/c},
\qquad c = 2 + c_0,
$$

with $c_0 > 0$ for boxy and $c_0 < 0$ for discy isophotes; $c_0 = 0$ gives
the Sérsic profile. The area within $m$ is $K(c)\,q\,m^2$, with
$K(c) = 4\,\Gamma(1 + 1/c)^2/\Gamma(1 + 2/c)$, so $R_e$ remains the
semi-major radius of the half-light isophote and the flux normalization is
analytic. Where $c$ is not an even integer, the radius is not smooth on the
axes of the isophotes, and those cells are integrated adaptively. At
$c_0 = -1$ it has a kink there, and the rendering is accurate to about
1e-6 of the flux instead of 1e-8.

| Parameter | Meaning | Units | Domain |
| --- | --- | --- | --- |
| `c0` | isophote shape | | $c_0 \ge -1$, convex isophotes |

and the parameters of the Sérsic profile.

**Python.** `baymorph.models.GeneralizedSersicComponent(flux, re, n, q, c0,
theta, x0, y0)`.

### Core-Sérsic profile

**Definition.** On ellipses of elliptical radius $R$,

$$
I(R) = I' \left[1 + \left(\frac{R_b}{R}\right)^{\alpha}\right]^{\gamma/\alpha}
\exp\left\{-b_n \left[\frac{R^{\alpha} + R_b^{\alpha}}{R_e^{\alpha}}
\right]^{1/(\alpha n)}\right\}
$$

(Graham et al. 2003; Trujillo et al. 2004): a power law of slope $\gamma$
within the break radius $R_b$, the Sérsic profile beyond it, and a
transition that sharpens with $\alpha$. $I'$ follows from `flux` by a
radial integral computed numerically in $\ln R$, with the power law added
analytically below $10^{-10} R_b$; it agrees with an independent integral
within 1e-9. The power-law cusp is integrated adaptively; it converges for
the validated slopes, and a slope near 2, where the central flux diverges,
raises an error instead.

| Parameter | Meaning | Units | Domain |
| --- | --- | --- | --- |
| `rb` | break radius | pixels | $R_b > 0$ |
| `gamma` | inner slope | | $\gamma < 2$ |
| `alpha` | sharpness of the transition | | $\alpha > 0$ |

and the parameters of the Sérsic profile.

**Python.** `baymorph.models.CoreSersicComponent(flux, re, n, rb, gamma,
alpha, q, theta, x0, y0)`.

### Broken exponential disc

**Definition.** On ellipses of elliptical radius $r$, the disc of Erwin,
Pohlen & Beckman (2008),

$$
I(r) = I_0 \exp\left(-\frac{r}{h_{\mathrm{in}}}\right)
\left[\frac{1 + e^{(r - r_b)/w_b}}{1 + e^{-r_b/w_b}}\right]^{
w_b (1/h_{\mathrm{in}} - 1/h_{\mathrm{out}})},
$$

with the width $w_b$ of the break in place of their sharpness
$\alpha = 1/w_b$, so that every parameter but the flux is a length. $I_0$
follows from `flux` by a radial integral computed numerically and split at
the break. With $h_{\mathrm{out}} = h_{\mathrm{in}}$ it is the exponential
disc, which it reproduces within 2.3e-7 of the peak.

| Parameter | Meaning | Units | Domain |
| --- | --- | --- | --- |
| `h_in`, `h_out` | scale lengths inside and outside the break | pixels | positive |
| `rb` | break radius | pixels | positive |
| `wb` | width of the break | pixels | positive |

and the flux, axis ratio, position angle and centre of the Sérsic profile.

**Python.** `baymorph.models.BrokenExponentialComponent(flux, h_in, h_out,
rb, wb, q, theta, x0, y0)`.

### Ferrers bar

**Definition.** On generalized ellipses of radius $m$ and shape $c_0$, the
Ferrers profile in the form of GALFIT (Peng et al. 2010) and Imfit (Erwin
2015), with its central slope fixed to zero,

$$
I(m) = I_0 \left[1 - \left(\frac{m}{a}\right)^{2}\right]^{\alpha}
\quad (m < a), \qquad I(m) = 0 \quad (m \ge a).
$$

The light ends at the semi-major length $a$; the slope at the edge is
infinite for $\alpha < 1$, finite for $\alpha = 1$ and zero beyond. Boxy
isophotes, $c_0 > 0$, describe bars (Athanassoula et al. 1990). The flux is
analytic,

$$
F = \frac{K(c)\, q\, I_0\, a^{2}}{\alpha + 1}.
$$

The cells within two cells of the edge, and the axes inside the bar for
isophotes that are not smooth there, are integrated adaptively; the cells
the edge crosses are split until their side is $a/512$. A bar inside the
image then holds its analytic flux within 1e-7.

| Parameter | Meaning | Units | Domain |
| --- | --- | --- | --- |
| `a` | semi-major length, where the light ends | pixels | $a > 0$ |
| `alpha` | sharpness of the edge | | $\alpha > 0$ |
| `c0` | isophote shape | | $c_0 \ge -1$, convex isophotes |

and the flux, axis ratio, position angle and centre of the Sérsic profile.

**Python.** `baymorph.models.FerrersBarComponent(flux, a, alpha, c0, q,
theta, x0, y0)`.

### Gaussian ring

**Definition.** On ellipses of elliptical radius $m$, the ring of Imfit
(Erwin 2015),

$$
I(m) = A \exp\left[-\frac{(m - r)^{2}}{2\sigma^{2}}\right],
$$

whose crest lies at the semi-major radius $r$ and whose width along the
major axis is $\sigma$. The flux is analytic,

$$
F = 2\pi q A \left\{\sigma^{2} e^{-r^{2}/(2\sigma^{2})}
+ \sqrt{\frac{\pi}{2}}\, r\sigma
\left[1 + \operatorname{erf}\left(\frac{r}{\sqrt{2}\,\sigma}\right)
\right]\right\},
$$

the circumference times the area of the Gaussian, $2\pi r q$ times
$\sqrt{2\pi}\,\sigma A$, for a ring much wider than its width. On the minor
axis the ring is $q\sigma$ wide; a ring narrower than one cell there is
integrated adaptively within $7\sigma$ of its crest.

| Parameter | Meaning | Units | Domain |
| --- | --- | --- | --- |
| `r` | semi-major radius of the crest | pixels | $r > 0$ |
| `sigma` | width along the major axis | pixels | $\sigma > 0$ |

and the flux, axis ratio, position angle and centre of the Sérsic profile.

**Python.** `baymorph.models.GaussianRingComponent(flux, r, sigma, q, theta,
x0, y0)`.
