# Bulge and Disc (`bulge_disk`)

A Sérsic bulge and an [exponential disc](COMPONENTS.md#exponential-disc)
with a shared centre: the classical bulge and disc decomposition.

## Definition

$$
I(\mathbf{x})
=
I_{\mathrm{Sersic}}(\mathbf{x};\, F_b, R_{e,b}, n_b, q_b, \theta_b, x_0, y_0)
+
I_{\mathrm{Sersic}}(\mathbf{x};\, F_d, R_{e,d}, n = 1, q_d, \theta_d, x_0, y_0).
$$

![Images of a Sérsic bulge, of an exponential disc and of their sum after the PSF, and their radial profiles](figures/bulge_disk_decomposition-light.png#gh-light-mode-only)
![Images of a Sérsic bulge, of an exponential disc and of their sum after the PSF, and their radial profiles](figures/bulge_disk_decomposition-dark.png#gh-dark-mode-only)

*A bulge of index 3.5 and an exponential disc with a bulge-to-total ratio of 0.3, after a Gaussian PSF of 2.5 px FWHM: the bulge dominates the centre, the disc the outskirts.*

## Parameters

| Parameter | Meaning | Units | Default prior |
| --- | --- | --- | --- |
| `flux_bulge` | total flux of the bulge | image units | none: must be declared |
| `re_bulge` | half-light radius of the bulge | pixels | uniform 1 to 30 |
| `n_bulge` | Sérsic index of the bulge | | uniform 1.5 to 6 |
| `q_bulge` | axis ratio of the bulge | | uniform 0.4 to 1 |
| `theta_bulge` | position angle of the bulge | radians | uniform −π/2 to π/2 |
| `flux_disk` | total flux of the disc | image units | none: must be declared |
| `re_disk` | half-light radius of the disc | pixels | uniform 10 to 120 |
| `q_disk` | axis ratio of the disc | | uniform 0.4 to 1 |
| `theta_disk` | position angle of the disc | radians | uniform −π/2 to π/2 |
| `x0`, `y0` | shared centre | pixels | uniform ±8 about the target seed |

`re_disk` is the half-light radius of the disc, not its exponential scale
length, which is `re_disk / 1.678`. The intrinsic model also gives the total
flux and the bulge-to-total ratio (`total_flux` and `bulge_to_total`).

## What it represents

A galaxy made of a spheroidal component, the bulge, with a free Sérsic
index, and a disc whose profile is exponential. Unlike
[`double_sersic`](DOUBLE_SERSIC.md), whose two components are general Sérsic
profiles, this model assigns a physical meaning to its components through
the fixed index of the disc. The shared centre belongs to the fitting
parameterization; the bulge and the disc keep separate axis ratios and
position angles. Bars, rings, spiral arms and a disc truncation are not
represented.

## Known degeneracies

- The bulge-to-total ratio is correlated with the index and the radius of
  the bulge: the bulge and the disc trade light at the radii where both
  contribute.
- Where the radius priors overlap, between 10 and 30 px, a compact disc and
  an extended bulge can describe similar light.
- Each component inherits the degeneracies of the
  [single Sérsic](SINGLE_SERSIC.md#known-degeneracies).

![Three plots: the bulge, disc and total profiles of two decompositions with bulge-to-total ratios 0.30 and 0.45, and the difference of their totals in percent](figures/bulge_disk_tradeoff-light.png#gh-light-mode-only)
![Three plots: the bulge, disc and total profiles of two decompositions with bulge-to-total ratios 0.30 and 0.45, and the difference of their totals in percent](figures/bulge_disk_tradeoff-dark.png#gh-dark-mode-only)

*Two decompositions of nearly the same galaxy: with B/T forced from 0.30 to 0.45, the bulge and disc radii and the bulge index adjust so that the totals agree within about 1 percent inside 8 px and within 8 percent out to 45 px, while the components differ widely.*

## Validation

- **Rendering.** The model renders as the exact sum of the Sérsic bulge and
  the exponential disc.
- **Recovery.** A controlled test recovers the posterior of the two fluxes of
  a simulated 25 x 25 px image with a Gaussian PSF, with the structural
  parameters fixed; the full structural recovery has not been validated.
- **Calibration.** No calibration campaign has been run for this model.

## Configuration

```yaml
fit:
  models:
    - bulge_disk
  priors:
    bulge_disk:
      flux_bulge:
        type: uniform
        lower: 100.0
        upper: 50000.0
      flux_disk:
        type: uniform
        lower: 100.0
        upper: 100000.0
```

## Python

`baymorph.models.BulgeDiskModel(bulge, disk)` combines a `SersicModel` bulge
and an `ExponentialComponent` disc;
`baymorph.fitting.make_bulge_disk_psf_model` builds the PSF-convolved model a
fit evaluates.
