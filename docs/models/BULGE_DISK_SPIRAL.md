# Bulge, Disc and Spiral Arms (`bulge_disk_spiral`)

A Sérsic bulge and an exponential disc with two logarithmic spiral arms
([spiral disc](COMPONENTS.md#exponential-disc-with-spiral-arms)), sharing a
centre. Available for fits of a single image.

## Definition

$$
I(\mathbf{x}) = I_{\mathrm{Sersic}}(\mathbf{x};\,F_b, R_{e,b}, n_b, q_b,
\theta_b) + I_{\mathrm{disc}}(\mathbf{x};\,F_d, R_{e,d}, q_d, \theta_d,
A_0, w, \varphi_0, r_{\mathrm{in}}),
$$

the bulge of [`bulge_disk`](BULGE_DISK.md) and its exponential disc
$I_d$ modulated by two arms,

$$
I_{\mathrm{disc}}(R, \varphi) = I_d(R) \left\{1 + A(R) \cos\left[
2(\varphi - \varphi_0) - 2 w \ln\left(\frac{R}{R_{e,d}}\right)\right]
\right\},
\qquad
A(R) = A_0 \left[1 - e^{-(R/r_{\mathrm{in}})^{2}}\right],
$$

where $R$ and $\varphi$ are the radius and the azimuth in the plane of the
disc, deprojected with $q_d$ and $\theta_d$. The crests of the arms follow
$\varphi = \varphi_0 + w \ln(R/R_{e,d})$ modulo $\pi$: $w = \cot\psi$ for
the pitch angle $\psi$, and $\varphi_0$ is the azimuth of an arm at
$R_{e,d}$, where the arms are best measured, which keeps $\varphi_0$ and $w$
nearly uncorrelated. The amplitude fades toward the centre, where the phase
diverges, over the radius $r_{\mathrm{in}}$.

The arms are the logarithmic-spiral mode with two arms that the
two-dimensional Fourier analyses of
[Puerari & Dottori (1992)][puerari-dottori1992] and
[Davis et al. (2012)][davis2012] measure on images of spiral galaxies.
GALFIT ([Peng et al. 2010][peng2010]) draws logarithmic arms differently,
by rotating the coordinates of a component. The fading $A(R)$ toward the
centre is a choice of BayMorph.

The cosine integrates to zero over every ellipse, so the disc holds the
flux $F_d$ exactly, and it is non-negative for $0 \le A_0 \le 1$. The model
without arms, $A_0 = 0$, is [`bulge_disk`](BULGE_DISK.md), which it then
reproduces bit for bit.

**Sense of winding.** The azimuth runs from the major axis toward the
$+$minor direction, which is counter-clockwise on the detector with $x$ to
the right and $y$ up. For $w > 0$ the arms wind counter-clockwise outward
and draw a Z; for $w < 0$ they draw an S. On an image displayed with $y$
down, or on a detector with inverted parity, the two letters swap.

## Parameters

| Parameter | Meaning | Units | Default prior |
| --- | --- | --- | --- |
| `flux_bulge` | total flux of the bulge | image units | none: must be declared |
| `re_bulge` | half-light radius of the bulge | pixels | uniform 1 to 30 |
| `n_bulge` | Sérsic index of the bulge | | uniform 1.5 to 6 |
| `q_bulge` | axis ratio of the bulge | | uniform 0.4 to 1 |
| `theta_bulge` | position angle of the bulge | radians | uniform −π/2 to π/2 |
| `flux_disk` | total flux of the disc, arms included | image units | none: must be declared |
| `re_disk` | half-light radius of the disc | pixels | uniform 10 to 120 |
| `q_disk` | axis ratio of the disc | | uniform 0.4 to 1 |
| `theta_disk` | position angle of the disc | radians | uniform −π/2 to π/2 |
| `amplitude_spiral` | relative amplitude $A_0$ of the arms | | uniform 0 to 1 |
| `winding_spiral` | $w = \cot\psi$: positive for a Z, negative for an S | | uniform −6 to 6 |
| `phase_spiral` | azimuth $\varphi_0$ of an arm at `re_disk`, in the plane of the disc | radians | uniform 0 to π |
| `r_in_spiral` | radius over which the arms fade toward the centre | pixels | uniform 1 to 60 |
| `x0`, `y0` | shared centre | pixels | uniform ±8 about the target seed |

The default priors are generic and in pixels; they must be checked against
the target and the image. `winding_spiral` must lie between −6 and 6, pitch
angles of at least 9.5°, and `re_disk`, `q_disk` and `r_in_spiral` must be
at least 10 px, 0.4 and 1 px: the domain in which the rendering of the arms
is validated, which configuration validation enforces. The pitch angle
$\psi = \operatorname{arccot} |w|$ is a derived quantity: under the uniform
prior in $w$, half of the prior mass lies at pitch angles below 18.4°.

![Images of three discs with two arms: arms that draw a Z, arms that draw an S, and tighter arms, with the crests drawn over them, and a plot of the pitch angle against the winding](figures/spiral_arms-light.png#gh-light-mode-only)
![Images of three discs with two arms: arms that draw a Z, arms that draw an S, and tighter arms, with the crests drawn over them, and a plot of the pitch angle against the winding](figures/spiral_arms-dark.png#gh-dark-mode-only)

*Face-on discs with R_e = 20 px, A₀ = 0.7, φ₀ = 0.4 and r_in = 4 px, shown with x to the right and y up. The lines are the crests φ = φ₀ + w ln(R/R_e) of the two arms. The last panel gives the pitch angle of the arms, ψ = arctan(1/|w|), against the winding.*

![Images of the convolved bulge, the disc with its arms and their sum, with the radial profiles of the three](figures/bulge_disk_spiral_decomposition-light.png#gh-light-mode-only)
![Images of the convolved bulge, the disc with its arms and their sum, with the radial profiles of the three](figures/bulge_disk_spiral_decomposition-dark.png#gh-dark-mode-only)

*A bulge of 25 percent of the light (n = 3, Rₑ = 3 px) and a disc of 75 percent (Rₑ = 16 px, q = 0.8) with arms of A₀ = 0.6 and w = 2.5, after a Gaussian PSF of 2.5 px FWHM. Averaged over circles, the arms vanish from the profile of the disc.*

## What it represents

A spiral galaxy with a bulge and two arms, described as a density wave on
its disc. The posterior of the pitch angle comes from a forward model with
the PSF and the noise, jointly with the bulge and the disc. The pitch angle
correlates with the mass of the central black hole
([Seigar et al. 2008][seigar2008];
[Davis, Graham & Seigar 2017][davis2017]) and with the Hubble type
([Kennicutt 1981][kennicutt1981]).

Real arms are not exact logarithmic spirals: their pitch varies from arm to
arm ([Lingard et al. 2021][lingard2021]), and flocculent arms, dust lanes and
star-forming regions, strongest in blue bands, leave residuals this model
does not describe.

## Known degeneracies

- At low signal-to-noise `amplitude_spiral` and `winding_spiral` are
  correlated: faint arms constrain their winding poorly.
- `r_in_spiral` and `amplitude_spiral` both set the light of the arms in the
  inner disc.
- `phase_spiral` is periodic with period π. A posterior that straddles 0
  and π is split between the two ends of its prior, and the summaries of
  the package, which are not circular, then misreport it, as they would a
  position angle near ±π/2.
- The bulge and the inner disc share the light at small radii, as in
  [`bulge_disk`](BULGE_DISK.md).

## Validation

- **Rendering.** Tested on 182 cases under PSFs of 2.3 and 3.5 px FWHM,
  with windings up to |w| = 11.4, a pitch angle of 5°, and axis ratios of
  0.6 and 0.4. Within the domain above the largest pixel error is 2.0e-3 of
  the peak and the largest flux error 9.3e-6.
- **Arms near the centre.** Where the arms wind faster than a pixel, near
  the centre, the rendering integrates them adaptively; the largest error
  there is 8e-7 of the peak of the arms.
- **Deep images.** The disc with its arms stays below the noise; a compact
  bright bulge sets the limit, as in
  [`bulge_broken_disk`](BULGE_BROKEN_DISK.md).
- **Flux.** The arms integrate to zero within 1e-8 of the flux of the disc,
  and with `amplitude_spiral` = 0 the model is exactly `bulge_disk`.
- **Sense of winding.** w > 0 winds counter-clockwise outward on the
  detector, drawing a Z; w < 0 winds clockwise, drawing an S.
- **Recovery.** On a simulated spiral galaxy, the posterior of
  `winding_spiral` and `phase_spiral` covers the true values.
- **Cost.** Two to three times the cost of `bulge_disk`.
- **Real data.** Not yet used on real images.

## Configuration

```yaml
fit:
  models:
    - bulge_disk_spiral
  priors:
    bulge_disk_spiral:
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

`baymorph.models.LogSpiralDiskComponent` is the disc with its arms;
`baymorph.fitting.make_bulge_disk_spiral_psf_model` builds the
PSF-convolved model a fit evaluates.

## References

- [Davis B. L., Graham A. W., Seigar M. S., 2017, MNRAS, 471, 2187][davis2017]: the pitch angle and the black-hole mass.
- [Davis B. L. et al., 2012, ApJS, 199, 33][davis2012]: logarithmic-spiral modes and the pitch angle.
- [Freeman K. C., 1970, ApJ, 160, 811][freeman1970]: the exponential disc.
- [Kennicutt R. C., 1981, AJ, 86, 1847][kennicutt1981]: the pitch angle along the Hubble sequence.
- [Lingard T. et al., 2021, MNRAS, 504, 3364][lingard2021]: the variation of pitch between arms.
- [Peng C. Y. et al., 2010, AJ, 139, 2097][peng2010]: logarithmic arms by a rotation of coordinates.
- [Puerari I., Dottori H. A., 1992, A&AS, 93, 469][puerari-dottori1992]: logarithmic-spiral modes of spiral galaxies.
- [Seigar M. S. et al., 2008, ApJ, 678, L93][seigar2008]: the pitch angle and the black-hole mass.
- [Sérsic J. L., 1968, Atlas de Galaxias Australes, Observatorio Astronómico, Córdoba][sersic1968]: the Sérsic profile of the bulge.

[davis2017]: https://ui.adsabs.harvard.edu/abs/2017MNRAS.471.2187D/abstract
[davis2012]: https://ui.adsabs.harvard.edu/abs/2012ApJS..199...33D/abstract
[freeman1970]: https://ui.adsabs.harvard.edu/abs/1970ApJ...160..811F/abstract
[kennicutt1981]: https://ui.adsabs.harvard.edu/abs/1981AJ.....86.1847K/abstract
[lingard2021]: https://ui.adsabs.harvard.edu/abs/2021MNRAS.504.3364L/abstract
[peng2010]: https://ui.adsabs.harvard.edu/abs/2010AJ....139.2097P/abstract
[puerari-dottori1992]: https://ui.adsabs.harvard.edu/abs/1992A%26AS...93..469P/abstract
[seigar2008]: https://ui.adsabs.harvard.edu/abs/2008ApJ...678L..93S/abstract
[sersic1968]: https://ui.adsabs.harvard.edu/abs/1968adga.book.....S/abstract
