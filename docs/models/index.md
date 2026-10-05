# Models

This section describes every model a BayMorph fit is built from: the
morphology of the galaxy, and the observation effects that turn it into the
expected image of the detector. Each page gives the definition, the
parameters with their units, domains and default priors, what the model
represents and what it does not, its known degeneracies, its validation
status, how to select it, and the references its model and formulas come
from, each linked to its record in NASA ADS.

## From a model to an expected image

The expected value of the pixel at position $\mathbf{x}$ is

$$
\mu(\mathbf{x})
=
\left[P \circledast I\right](\mathbf{x})
+
B(\mathbf{x}),
$$

where $I$ is the intrinsic surface brightness of the galaxy, the sum of its
morphological components, $P$ is the point-spread function of the
observation, and $B$ is the residual background. The morphology describes
the source; the PSF, the pixel response and the background describe the
measurement, and never change the meaning of a morphological parameter.

![Four images: an intrinsic Sérsic galaxy, the same galaxy convolved with the PSF, plus a residual background plane, and with noise added](figures/forward_model-light.png#gh-light-mode-only)
![Four images: an intrinsic Sérsic galaxy, the same galaxy convolved with the PSF, plus a residual background plane, and with noise added](figures/forward_model-dark.png#gh-dark-mode-only)

*In order, from (1) to (4): the intrinsic profile, its convolution with a Gaussian PSF of 5 px FWHM, the residual background added, and an observation with noise. The colour scale is logarithmic and shared by the four panels.*

| Part | Models | Page |
| --- | --- | --- |
| Morphological components | nine components: Sérsic, exponential disc, point source, generalized Sérsic, core-Sérsic, broken exponential disc, Ferrers bar, Gaussian ring, exponential disc with spiral arms | [Components](COMPONENTS.md) |
| Galaxy models | `single_sersic`, `double_sersic`, `sersic_point_source`, `bulge_disk`; `boxy_sersic`, `core_sersic`, `bulge_broken_disk`, `bulge_disk_bar`, `bulge_disk_ring`, `bulge_disk_spiral` | one page each, below |
| Point-spread function | Gaussian, empirical | [Point-Spread Functions](PSF.md) |
| Noise | correlated Gaussian; experimental Poisson likelihoods | [Noise Models](NOISE.md) |
| Sky and residual background | sky subtraction; constant, plane | [Sky and Background](BACKGROUND.md) |

A new model of any kind gets its own page in the matching group, with the
same structure: definition, parameters, what it represents, degeneracies,
validation, configuration and references.

## The galaxy models

| Model | Components | Sampled morphology parameters | Use |
| --- | --- | --- | --- |
| [`single_sersic`](SINGLE_SERSIC.md) | one Sérsic | 7 | the baseline description of a galaxy |
| [`double_sersic`](DOUBLE_SERSIC.md) | two labelled Sérsic components, shared centre | 12 | a galaxy whose profile one Sérsic cannot describe |
| [`sersic_point_source`](SERSIC_POINT_SOURCE.md) | Sérsic and unresolved source, shared centre | 8 | a galaxy with an unresolved nucleus |
| [`bulge_disk`](BULGE_DISK.md) | Sérsic bulge and exponential disc, shared centre | 11 | a bulge and disc decomposition |
| [`boxy_sersic`](BOXY_SERSIC.md) | one Sérsic on generalized ellipses | 8 | boxy or discy isophotes |
| [`core_sersic`](CORE_SERSIC.md) | one core-Sérsic | 10 | a resolved depleted core |
| [`bulge_broken_disk`](BULGE_BROKEN_DISK.md) | Sérsic bulge and broken exponential disc, shared centre | 14 | a disc of type I, II or III |
| [`bulge_disk_bar`](BULGE_DISK_BAR.md) | Sérsic bulge, exponential disc and Ferrers bar, shared centre | 17 | a barred disc galaxy |
| [`bulge_disk_ring`](BULGE_DISK_RING.md) | Sérsic bulge, exponential disc and Gaussian ring, shared centre | 16 | a disc galaxy with a ring |
| [`bulge_disk_spiral`](BULGE_DISK_SPIRAL.md) | Sérsic bulge and exponential disc with two spiral arms, shared centre | 15 | a two-armed spiral galaxy |

The last six are available for fits of a single image: a joint fit of
several images does not accept them, and their priors must stay within the
domain in which their rendering is validated, which each page states.

A fit samples these parameters together with those of the residual
background: one for a constant background, three for a plane.

![Grid of the ten morphology models against the nine components: a dot where a model uses a component, two for the two Sérsic components of double_sersic; the first four models accept a single image or several, the last six a single image only](figures/model_composition-light.png#gh-light-mode-only)
![Grid of the ten morphology models against the nine components: a dot where a model uses a component, two for the two Sérsic components of double_sersic; the first four models accept a single image or several, the last six a single image only](figures/model_composition-dark.png#gh-dark-mode-only)

*Which components each model is made of, and how every model becomes an expected image.*

## Conventions

These conventions hold for every model.

- **Coordinates.** Positions are in pixels of the prepared cutout. The
  pixel with index `[j, i]` is centred at `(x, y) = (i, j)` and covers
  `x − 0.5` to `x + 0.5`; scientific coordinates are written `(x, y)`, NumPy
  arrays are indexed `[y, x]`. The default priors of `x0` and `y0` are
  centred on the target seed, transformed to cutout coordinates.
- **Radii.** `re` is the half-light radius measured along the major axis, in
  pixels.
- **Axis ratio.** `q = b / a`, with `0 < q ≤ 1`.
- **Position angle.** `theta` is the angle of the major axis, in radians,
  measured from the `+x` axis towards `+y`. It has period π, and it is
  undefined for a round component, `q = 1`.
- **Flux.** `flux` is the total flux of the continuous component over the
  full plane, in the units of the image. A finite image holds only the part
  of it that falls inside the image.

## Choosing models

The models to fit are listed in `fit.models`; each is fitted in its own fit
directory. Every flux needs a declared prior, and the other parameters have
the default priors of the
[Configuration reference](../reference/CONFIGURATION.md#priors).
`fit.comparison.enabled` computes the Bayes factor of `double_sersic` over
`single_sersic` when both are fitted. BayMorph ranks models; it never
selects one for the user.
