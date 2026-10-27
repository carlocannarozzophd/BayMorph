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
| `psf.empirical.construction.sky_inner_radius`, `sky_outer_radius` | 80, 160 | The ring where the sky of each star is measured, in pixels. |

Every option is in the [Configuration reference](../reference/CONFIGURATION.md#psf).

## Files

`build_psf()` writes `psf/psf.fits` and `psf/psf_metadata.json`, the
selected stars in `psf/psf_candidates.json`, and two diagnostic figures in
`psf/diagnostics/`: the stamps of the stars and the profile of the kernel.

## Limits

- One kernel describes the whole image: a PSF that changes across the field
  is not modelled.
- Light beyond 64 pixels from the centre of a star is not in the kernel.
- The kernel was validated on fields with stars at least 50 pixels apart;
  a crowded field may need a stricter selection.

The [Point-Spread Functions](../models/PSF.md) page describes both PSF
models and how a galaxy is rendered with them.
