# Sérsic plus Point Source (`sersic_point_source`)

An extended [Sérsic profile](COMPONENTS.md#sersic-profile) with an
unresolved [point source](COMPONENTS.md#point-source) at the same centre.

## Definition

$$
I(\mathbf{x})
=
I_{\mathrm{Sersic}}(\mathbf{x};\, F_S, R_e, n, q, \theta, x_0, y_0)
+
F_{PS}\,\delta(\mathbf{x} - \mathbf{x}_0).
$$

The Sérsic component is convolved with the PSF; the point source is the PSF
itself, scaled by its flux and placed at the shared centre, and is never
convolved twice.

![Images of a Sérsic component, of an unresolved source and of their sum after the PSF, and their radial profiles](figures/sersic_point_source_decomposition-light.png#gh-light-mode-only)
![Images of a Sérsic component, of an unresolved source and of their sum after the PSF, and their radial profiles](figures/sersic_point_source_decomposition-dark.png#gh-dark-mode-only)

*A Sérsic profile with a point source of one fifth of the total flux at its centre, after a Gaussian PSF of 2.5 px FWHM: the point source adds light only within a few pixels of the centre.*

## Parameters

| Parameter | Meaning | Units | Default prior |
| --- | --- | --- | --- |
| `flux_sersic` | total flux of the extended component | image units | none: must be declared |
| `re` | its half-light radius along the major axis | pixels | uniform 1 to 120 |
| `n` | its Sérsic index | | uniform 0.5 to 6 |
| `q` | its axis ratio | | uniform 0.4 to 1 |
| `theta` | its position angle | radians | uniform −π/2 to π/2 |
| `flux_point_source` | total flux of the unresolved source | image units | none: must be declared |
| `x0`, `y0` | shared centre | pixels | uniform ±8 about the target seed |

The intrinsic model also gives the total flux and the fraction of it in the
point source (`total_flux` and `point_source_fraction`).

## What it represents

A galaxy with an unresolved central component, such as a compact nucleus or
an active nucleus, whose light the PSF fully describes. The shared centre is
a modelling assumption, not a property of either component. The point source
has no size: a central component that the PSF resolves is not a point
source.

## Known degeneracies

- The point source and the core of the Sérsic profile compete for the central
  light: a high index or a small radius concentrates the Sérsic light like a
  point source, so `flux_point_source` is correlated with `n` and `re`.
- An error of the PSF core goes into the point-source flux and the Sérsic
  index; the [PSF](PSF.md) must be accurate in its core for the split to be
  meaningful.
- The extended component inherits the degeneracies of the
  [single Sérsic](SINGLE_SERSIC.md#known-degeneracies).

## Validation

- **Rendering.** The point source is placed through the sub-pixel kernels of
  an empirical PSF (AD-046), or through a shifted Gaussian kernel.
- **Recovery.** A controlled test fits a simulated 21 x 21 px image with a
  Gaussian PSF and recovers the Sérsic flux, its radius and the point-source
  flux, with the other parameters fixed.
- **Calibration.** No calibration campaign has been run for this model.

## Configuration

```yaml
fit:
  models:
    - sersic_point_source
  priors:
    sersic_point_source:
      flux_sersic:
        type: uniform
        lower: 1000.0
        upper: 100000.0
      flux_point_source:
        type: uniform
        lower: 10.0
        upper: 10000.0
```

## Python

`baymorph.models.SersicPointSourceModel(sersic, point_source)` combines a
`SersicModel` and a `PointSourceModel` with the same centre;
`baymorph.fitting.make_sersic_point_source_psf_model` builds the observed
model a fit evaluates.
