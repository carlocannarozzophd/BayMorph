# Double Sérsic (`double_sersic`)

Two [Sérsic profiles](COMPONENTS.md#sersic-profile) with a shared centre,
each with its own flux, radius, index, axis ratio and position angle.

## Definition

$$
I(\mathbf{x})
=
I_{\mathrm{Sersic}}(\mathbf{x};\, F_b, R_{e,b}, n_b, q_b, \theta_b, x_0, y_0)
+
I_{\mathrm{Sersic}}(\mathbf{x};\, F_d, R_{e,d}, n_d, q_d, \theta_d, x_0, y_0).
$$

![Images of the two Sérsic components and of their sum after the PSF, and their radial profiles](figures/double_sersic_decomposition-light.png#gh-light-mode-only)
![Images of the two Sérsic components and of their sum after the PSF, and their radial profiles](figures/double_sersic_decomposition-dark.png#gh-dark-mode-only)

*The two labelled components, a compact one of index 4 and an extended one of index 1, and their sum after a Gaussian PSF of 2.5 px FWHM; right, the mean intensity in circular annuli.*

## Parameters

| Parameter | Meaning | Units | Default prior |
| --- | --- | --- | --- |
| `flux_bulge` | total flux of the first component | image units | none: must be declared |
| `re_bulge` | its half-light radius | pixels | uniform 1 to 30 |
| `n_bulge` | its Sérsic index | | uniform 1.5 to 6 |
| `q_bulge` | its axis ratio | | uniform 0.4 to 1, from the `q` default |
| `theta_bulge` | its position angle | radians | uniform −π/2 to π/2, from the `theta` default |
| `flux_disk` | total flux of the second component | image units | none: must be declared |
| `re_disk` | its half-light radius | pixels | uniform 10 to 120 |
| `n_disk` | its Sérsic index | | uniform 0.5 to 3 |
| `q_disk` | its axis ratio | | uniform 0.4 to 1, from the `q` default |
| `theta_disk` | its position angle | radians | uniform −π/2 to π/2, from the `theta` default |
| `x0`, `y0` | shared centre | pixels | uniform ±8 about the target seed |

A prior declared under `q` or `theta` applies to both components unless a
component-specific prior is declared; the two components keep separate
parameters.

## What it represents

Two general Sérsic components of one galaxy, sharing a centre. The names
`bulge` and `disk` are labels of the components, defined by their different
default priors, a compact component of higher index and an extended one of
lower index; the model does not impose a physical bulge and disc
decomposition, since both components are general Sérsic profiles with free
indices. For a Sérsic bulge with an exponential disc, use
[`bulge_disk`](BULGE_DISK.md). The components share their centre, not their
shape.

## Known degeneracies

- Where the priors of the two components overlap, for example radii between
  10 and 30 px and indices between 1.5 and 3, the posterior has modes
  related by exchanging the labels.
- The split of the flux between the components is correlated with their
  radii and indices, and the posterior is far from Gaussian: a mixture of
  Student-t distributions about its modes matches it poorly
  ([model-informed evidence](../research/STRUCTURE_INFORMED_EVIDENCE_2026-09-29.md)).
  Its evidence therefore relies on a nested sampler such as Nautilus.
- Each component inherits the degeneracies of the
  [single Sérsic](SINGLE_SERSIC.md#known-degeneracies).

## Validation

- **Rendering.** The model renders as the exact sum of its two components,
  with the rendering of the single Sérsic.
- **Model comparison.** With `fit.comparison.enabled`, BayMorph computes the
  Bayes factor of `double_sersic` over `single_sersic` on the same prepared
  data, mask and likelihood.
- **Recovery and calibration.** No dedicated recovery or calibration
  campaign has been run for this model.

## Configuration

```yaml
fit:
  models:
    - single_sersic
    - double_sersic
  comparison:
    enabled: true
  priors:
    double_sersic:
      flux_bulge:
        type: uniform
        lower: 100.0
        upper: 50000.0
      flux_disk:
        type: uniform
        lower: 100.0
        upper: 100000.0
```

The flux prior of `single_sersic` must be declared as well; see
[Single Sérsic](SINGLE_SERSIC.md#configuration).

## Python

`baymorph.DoubleSersicModel(bulge, disk)` combines two `SersicModel`
components; `baymorph.fitting.make_double_sersic_psf_model` builds the
PSF-convolved model a fit evaluates.
