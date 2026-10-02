# Core-Sérsic (`core_sersic`)

A Sérsic profile whose centre is depleted: inside a break radius the light
follows a shallow power law instead of the inward extrapolation of the
Sérsic profile ([Graham et al. 2003][graham2003];
[Trujillo et al. 2004][trujillo2004]). One of the
families of AD-048: available for single-observation fits.

## Definition

$$
I(R) = I' \left[1 + \left(\frac{R_b}{R}\right)^{\alpha}\right]^{\gamma/\alpha}
\exp\left\{-b_n \left[\frac{R^{\alpha} + R_b^{\alpha}}{R_e^{\alpha}}
\right]^{1/(\alpha n)}\right\},
$$

on ellipses of axis ratio $q$, where $R$ is the elliptical radius. Within
the break radius $R_b$ the profile is the power law $R^{-\gamma}$; beyond
it, the Sérsic profile of index $n$ whose half-light radius is $R_e$; the
transition sharpens as $\alpha$ grows. $b_n$ is the Sérsic constant of
index $n$, so $R_e$ is the half-light radius of the outer Sérsic profile;
the half-light radius of the whole profile is a derived quantity, smaller by
the light the core lacks. `flux` is the total flux over the whole plane, and
$I'$ follows from it by a radial integral computed numerically.

## Parameters

| Parameter | Meaning | Units | Default prior |
| --- | --- | --- | --- |
| `flux` | total flux of the profile | image units | none: must be declared |
| `re` | half-light radius of the outer Sérsic profile | pixels | uniform 1 to 120 |
| `n` | Sérsic index of the outer profile | | uniform 0.5 to 6 |
| `rb` | break radius | pixels | uniform 0.1 to 20 |
| `gamma` | slope of the inner power law | | uniform 0 to 0.9 |
| `alpha` | sharpness of the transition | | uniform 0.5 to 2 |
| `q` | axis ratio | | uniform 0.4 to 1 |
| `theta` | position angle of the major axis | radians | uniform −π/2 to π/2 |
| `x0`, `y0` | centre | pixels | uniform ±8 about the target seed |

The default priors are generic and in pixels; they must be checked against
the target and the image.

- **`alpha`.** [Baes (2026)][baes2026] finds that the deprojected density of a
  core-Sérsic profile is monotonic only for `alpha` up to a critical value,
  which equals 2 when `gamma = 0` and is never below 2. The default bound
  of 2 is therefore physically admissible for every core.
- **`gamma`.** Depleted cores have inner slopes of 0.3 or less
  ([Dullo & Graham 2012][dullo-graham2012]). The bound of 0.9 is the
  validated domain of the rendering:
  steeper cusps, up to the limit of 2 at which the central flux diverges,
  are not resolved by the cell rules, which then raise an error.

![Two log-log plots of the core-Sérsic profile against radius in units of the break radius: inner slopes 0, 0.3 and 0.9, and transitions 0.5, 1 and 2, each with the Sérsic body for comparison](figures/core_sersic_profiles-light.png#gh-light-mode-only)
![Two log-log plots of the core-Sérsic profile against radius in units of the break radius: inner slopes 0, 0.3 and 0.9, and transitions 0.5, 1 and 2, each with the Sérsic body for comparison](figures/core_sersic_profiles-dark.png#gh-dark-mode-only)

*The core-Sérsic profile in units of $I'$, with $R_e$ = 20 px, n = 4 and $R_b$ = 2 px, against the inward extrapolation of its Sérsic body (dashed). Left: the inner slope; right: the sharpness of the transition, which for small α broadens it far beyond the break radius.*

## What it represents

The partially depleted core of a luminous early-type galaxy, the light
deficit that binary black holes are thought to scour, together with its
Sérsic body. Every elliptical brighter than M_V = −21.66 in the Virgo sample
of [Kormendy et al. (2009)][kormendy2009] has such a core.

![Images of a convolved core-Sérsic galaxy, of its Sérsic body and of the light the core lacks, with the radial profiles of the three](figures/core_sersic_deficit-light.png#gh-light-mode-only)
![Images of a convolved core-Sérsic galaxy, of its Sérsic body and of the light the core lacks, with the radial profiles of the three](figures/core_sersic_deficit-dark.png#gh-dark-mode-only)

*A core-Sérsic galaxy (n = 4, Rₑ = 20 px, R_b = 4 px, γ = 0.1, α = 2, q = 0.8) and its Sérsic body, the Sérsic profile it follows beyond the break, after a Gaussian PSF of 2.5 px FWHM. The core lacks 14 percent of the light of the body, within a few break radii; the vertical line marks the break radius.*

Cores measure 0.02 to 0.5 kpc ([Dullo 2019][dullo2019]). A break radius
smaller than the core of the PSF is not resolved: its posterior then
follows the prior, and
an evidence comparison with `single_sersic` depends on that prior.

## Known degeneracies

- `rb`, `gamma` and `alpha` are strongly coupled when the break is near the
  resolution: a smaller core with a steeper slope and a sharper transition
  can match a larger, flatter and smoother one.
- `re` and `n` describe the outer profile, and keep the degeneracies of
  [`single_sersic`](SINGLE_SERSIC.md).
- `theta` has period π and is undefined for `q = 1`.

## Validation

- **Rendering.** On the bench of AD-048, with the criteria of AD-046, the
  model meets every criterion on 102 cases with rb from 0.5 to 5 px, gamma
  from 0 to 0.9, alpha from 0.5 to 2 and n of 4 and 6, under effective PSFs
  of 2.34 and 3.5 px FWHM. The largest pixel difference is 5.4e-4 of the
  peak, and the target-like chi-square at most 0.053
  ([note](../research/MORPHOLOGY_FAMILIES_RENDERING_2026-10-02.md)).
- **Flux.** The normalization agrees with an independent radial integral
  within 1e-9.
- **Recovery.** On a simulated image whose core is resolved, the posterior
  of `rb` and `gamma` covers the true values
  (`tests/test_core_sersic_recovery.py`).
- **Cost.** A rendering of 1013 × 1013 px with an effective PSF of 27 px
  takes 74 ms, against 40 ms for `single_sersic`.
- **Real data.** No fit on real data has been made with this model.

## Configuration

```yaml
fit:
  models:
    - core_sersic
  priors:
    core_sersic:
      flux:
        type: uniform
        lower: 1000.0
        upper: 100000.0
      rb:
        type: uniform
        lower: 0.5
        upper: 10.0
```

Priors not declared take the defaults above; the values here are an
example, to be chosen for each image. A joint multi-observation fit rejects
this model before writing any output (AD-048).

## Python

`baymorph.models.CoreSersicComponent` is the intrinsic profile;
`baymorph.fitting.make_core_sersic_psf_model` builds the PSF-convolved
model a fit evaluates.

## References

- [Baes M., 2026, A&A, 709, A24][baes2026]: the physical limit on α.
- [Dullo B. T., 2019, ApJ, 886, 80][dullo2019]: the sizes of depleted cores.
- [Dullo B. T., Graham A. W., 2012, ApJ, 755, 163][dullo-graham2012]: the inner slopes of depleted cores.
- [Graham A. W. et al., 2003, AJ, 125, 2951][graham2003]: the core-Sérsic profile.
- [Kormendy J. et al., 2009, ApJS, 182, 216][kormendy2009]: the cores of Virgo ellipticals.
- [Trujillo I. et al., 2004, AJ, 127, 1917][trujillo2004]: the core-Sérsic profile.

[baes2026]: https://ui.adsabs.harvard.edu/abs/2026A%26A...709A..24B/abstract
[dullo2019]: https://ui.adsabs.harvard.edu/abs/2019ApJ...886...80D/abstract
[dullo-graham2012]: https://ui.adsabs.harvard.edu/abs/2012ApJ...755..163D/abstract
[graham2003]: https://ui.adsabs.harvard.edu/abs/2003AJ....125.2951G/abstract
[kormendy2009]: https://ui.adsabs.harvard.edu/abs/2009ApJS..182..216K/abstract
[trujillo2004]: https://ui.adsabs.harvard.edu/abs/2004AJ....127.1917T/abstract
