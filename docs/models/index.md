# Models

This section describes every model a BayMorph fit is built from: the
morphology of the galaxy, and the observation effects that turn it into the
expected image of the detector. Each page gives the definition, the
parameters with their units, domains and default priors, what the model
represents and what it does not, its known degeneracies, its validation
status, and how to select it.

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

*From left to right: the intrinsic profile, its convolution with a Gaussian PSF of 5 px FWHM, the residual background added, and an observation with noise. The colour scale is logarithmic and shared by the four panels.*

| Part | Models | Page |
| --- | --- | --- |
| Morphological components | Sérsic, exponential disc, point source | [Components](COMPONENTS.md) |
| Morphology models | `single_sersic`, `double_sersic`, `sersic_point_source`, `bulge_disk` | one page each, below |
| Point-spread function | Gaussian, empirical | [Point-Spread Functions](PSF.md) |
| Residual background | constant, plane | [Residual Background](BACKGROUND.md) |

## The morphology models

| Model | Components | Sampled morphology parameters | Use |
| --- | --- | --- | --- |
| [`single_sersic`](SINGLE_SERSIC.md) | one Sérsic | 7 | the baseline description of a galaxy |
| [`double_sersic`](DOUBLE_SERSIC.md) | two labelled Sérsic components, shared centre | 12 | a galaxy whose profile one Sérsic cannot describe |
| [`sersic_point_source`](SERSIC_POINT_SOURCE.md) | Sérsic and unresolved source, shared centre | 8 | a galaxy with an unresolved nucleus |
| [`bulge_disk`](BULGE_DISK.md) | Sérsic bulge and exponential disc, shared centre | 11 | a bulge and disc decomposition |

A fit samples these parameters together with those of the residual
background: one for a constant background, three for a plane.

![Diagram: the Sérsic, exponential-disc and point-source components feed the four morphology models, which are convolved with the PSF and summed with the residual background](figures/model_composition-light.png#gh-light-mode-only)
![Diagram: the Sérsic, exponential-disc and point-source components feed the four morphology models, which are convolved with the PSF and summed with the residual background](figures/model_composition-dark.png#gh-dark-mode-only)

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
  of it that falls inside the image. This is the implementation, as the
  status note under AD-028 records; the reconciliation of AD-028 with it is
  an open decision.

## Choosing models

The models to fit are listed in `fit.models`; each is fitted in its own fit
directory. Every flux needs a declared prior, and the other parameters have
the default priors of the
[Configuration Reference](../guide/CONFIGURATION.md#priors).
`fit.comparison.enabled` computes the Bayes factor of `double_sersic` over
`single_sersic` when both are fitted. BayMorph ranks models; it never
selects one for the user.

The scientific contract that these models implement, including the
distinction between a component and a probabilistic model, is in
[Morphological Components](../architecture/MORPHOLOGICAL_COMPONENTS.md).
