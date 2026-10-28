# Point-Spread Function

The point-spread function (PSF) describes how the telescope, the
atmosphere and the detector blur a point of light. BayMorph convolves every
model with it before comparing the model with the image, so the PSF must
describe your image well: an error in the PSF becomes an error in the size
and concentration of the galaxy.

=== "Empirical (default)"

    ```yaml
    psf:
      type: empirical
    ```

    Built from isolated stars of the same image. This is the choice for a
    real image.

=== "Gaussian"

    ```yaml
    psf:
      type: gaussian
      gaussian:
        fwhm: 2.5        # pixels
    ```

    An analytic Gaussian. Useful for simulations and quick tests; a real
    PSF has wings that a Gaussian does not have.

## Building an empirical PSF

```python
runner = BayMorphRunner("configs/galaxy.yaml")
runner.prepare_image()

candidates = runner.select_psf_candidates()
psf = runner.build_psf()
```

**Selecting the stars.** `select_psf_candidates()` keeps up to 30 sources
that look like isolated stars: compact, round, away from the image edges
and from other sources, with a peak between `min_peak` and `max_peak`, so
that saturated stars are left out. Each candidate is a dictionary with its
position, peak, flux and shape:

```python
len(candidates)              # 30
candidates[0]["x"], candidates[0]["y"], candidates[0]["max_value"]
```

**Screening the stars.** Shape alone does not tell a star from a compact
galaxy, a cosmic ray or a hot pixel, and the check of the kernel's centroid
does not see them either. So every candidate that passes the shape cuts is
measured on its own stamp: its size, the second moment of its light within
a Gaussian window matched to it, which does not grow with its brightness,
and the shape of its profile. The stars are the smallest group of six or
more candidates of one size, within 5 per cent, provided that the group is
narrow and stands out from the sizes next to it. Sharper sources (cosmic
rays, hot pixels), larger ones (galaxies) and sources unlike the group are
rejected, and the next candidates take their places. Without such a group,
or with two, the selection stops, saves no candidate, and says why and what
can help; the error holds the measurement and verdict of every candidate
(`error.record` in Python), and `baymorph` prints an account of them.

**Building the kernel.** `build_psf()` combines the candidates whose peak
is at least 30 times the noise into one kernel of 129 x 129 pixels,
normalized to unit sum. The stars fall at different
positions within their pixels, and BayMorph uses this to describe the PSF
on a grid eleven times finer than the pixels, which the fit needs for
compact galaxies. The sky around each star is measured in a ring 80 to 160
pixels from it, and the faint wings of the PSF are continued to the edge of
the kernel from the profile of the stars.

```python
psf.shape          # (129, 129)
psf.sum()          # 1.0

from baymorph.plotting import plot_log_image

fig, ax = plot_log_image(psf, title="Empirical PSF", show=True)
```

A kernel that is not finite, non-negative, normalized and centred is
rejected, never repaired.

## The settings to check

| Setting | Default | Check |
| --- | --- | --- |
| `psf.empirical.candidate_selection.max_peak` | 80000 | Set it below the saturation level of your image, in image units: the default is not scaled to it. |
| `psf.empirical.candidate_selection.min_peak` | 20 | Not scaled to your image either; the Advisor proposes 30 times the noise sigma of the image. |
| `psf.empirical.candidate_selection.min_peak_snr` | 30 | Lower it only if the image has too few bright stars. |
| `psf.empirical.candidate_selection.galaxy_exclusion_box` | none | A box where candidates are refused, for example around a large galaxy. |
| `psf.empirical.candidate_selection.point_source_screen` | `size_sequence_v1` | `none` selects the stars by shape only, as earlier releases did. |
| `psf.empirical.construction.sky_inner_radius`, `sky_outer_radius` | 80, 160 | The ring where the sky of each star is measured, in pixels. |

Every option is in the [Configuration reference](../reference/CONFIGURATION.md#psf).

## Files

`build_psf()` writes `psf/psf.fits` and `psf/psf_metadata.json`, the
selected stars in `psf/psf_candidates.json`, and two diagnostic figures in
`psf/diagnostics/`: the stamps of the stars and the profile of the kernel.
The screen of the stars is recorded in `psf/psf_candidate_screen.json`.

## Limits

- One kernel describes the whole image: a PSF that changes across the field
  is not modelled.
- The screen needs at least six stars with a peak signal-to-noise of 30,
  and they must stand out from the sources just larger than them. Barely
  resolved galaxies, 5 to 16 per cent larger than the stars, stop it in
  most synthetic fields once they number half as many as the stars: with
  3 of them next to the stars the group needs 11 stars, with 4 it needs 12,
  and the message of the stop gives the count. Expect stops in deep fields,
  where such galaxies are many.
- It stops when the size of the PSF changes by more than about 4 per cent
  across the region searched, and when six or more sources smaller than
  the stars but above the floor of 0.6 pixels in size, such as cosmic rays
  smoothed by resampling, lie below them.
- On a PSF sampled with little more than two pixels per FWHM and not
  resampled, the profile of a bright star changes with its position within
  a pixel, and the screen can stop although every candidate is a star.
- Galaxies a few per cent larger than the PSF can be kept, at a low weight.
- Without stars, a crowd of barely resolved galaxies of nearly one size can
  be taken for them. In synthetic fields with at most three stars among 60
  compact galaxies whose median size is 3 to 10 per cent above that of the
  PSF, the screen returned a list in 25 of 48, with kernels up to 11 per
  cent too broad, and every one of them passed the check of the kernel's
  centroid. Check the size and the scatter of the stellar sequence that the
  Advisor reports: in those fields, sequences of stars of one PSF scattered
  by at most 0.010 in `ln s`, crowds of galaxies by 0.016 or more.
- The screen keeps galaxies out of the kernel; it does not make the check
  of the kernel's centroid pass. With few stars, one of which dominates,
  or with a PSF whose light is not symmetric about its peak, such as a JWST
  PSF, the kernel can fail that check although every star is a point
  source.
- Light beyond 64 pixels from the centre of a star is not in the kernel.
- The kernel was validated on fields with stars at least 50 pixels apart;
  a crowded field may need a stricter selection.

The [Point-Spread Functions](../models/PSF.md) page describes both PSF
models and how a galaxy is rendered with them.
