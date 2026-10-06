# Masks

Other sources in the cutout, such as neighbouring galaxies, stars or
artefacts, must not enter the fit. BayMorph excludes them with a mask: a
boolean image in which `True` means *excluded* and `False` means *used by the
likelihood*.

## The automatic mask

During `prepare_image()`, BayMorph detects the sources of the cutout with a
segmentation map ([Photutils](https://photutils.readthedocs.io/)): on a
smoothed copy of the cutout, pixels more than `n_sigma` noise units above
the background, in groups of at least `n_pixels` connected pixels, are
sources, and touching sources are separated by deblending with `n_levels`
levels. The source that contains the target position is the target.

```yaml
preparation:
  detection:
    n_sigma: 3.0       # detection threshold, in noise units
    n_pixels: 20       # smallest source, in connected pixels
    n_levels: 32       # deblending levels
```

Not every other detected segment is masked. Over the wings of the target,
the light of the target lifts peaks of the noise above the threshold, and
each would become a segment of its own; masking them removes positive noise
from the fit and biases the flux, the effective radius and the Sérsic index
low. A segment is therefore masked only when it still holds a source once
the threshold is raised by the smooth light of the target: at least
`n_pixels` connected pixels above the median profile of the target, in
elliptical annuli, by the same margin that the detection asks above the
sky. Far from the target this is the detection itself. The other segments
are released and enter the fit like every other pixel.

```yaml
preparation:
  masking:
    contaminant_rule: local_excess   # or all_segments, which masks every segment
```

`all_segments` masks every detected segment, as earlier versions of
BayMorph did. It is kept so that products prepared before the rule can
still be used: a preparation records its rule, and products prepared under
the other rule are not reused. The rule decides segment by segment: a
segment that holds a neighbour and, through the deblending, a part of the
envelope of the target stays masked whole.

A detected segment ends where the smoothed image falls below the detection
threshold, but the faint outskirts of the source extend beyond it. Left in
the fit, the outskirts of many neighbours raise the fitted background, which
then takes light from the galaxy: on simulated images of a galaxy among many
fainter neighbours, keeping them in the fit lowered its flux, effective
radius and Sérsic index by 1 to 5 percent. BayMorph therefore grows every
masked segment by 20 pixels: each pixel within that distance of a masked
segment is excluded from the fit and from the measurement of the sky. On
the same simulations the growth kept those biases within 0.2 percent, and
within 3 percent when the sky also held structure on large scales. The
target, its geometry and the plateau are still measured with the segments
as detected.

```yaml
preparation:
  masking:
    contaminant_growth_pixels: 20    # the default, in pixels; 0 keeps the segments
```

The growth removes pixels: on a crowded image it can exclude half of the
cutout, including pixels close to the galaxy, so look at the mask. The
summary of the run, `runtime_report.yaml`, records as
`excluded_target_fraction` the fraction of the detected target that the
mask excludes. The 20 pixels were validated on images whose pixels sample a
seeing of about 2.3 pixels; on an image of a very different sampling, or
with a neighbour close to the target, a different growth may suit it better,
but `field_separated` (see [Noise Models](../models/NOISE.md)) requires 20.
The growth applies to the detected segments only: with `mask_mode: exact`,
the external mask is used as drawn and nothing is grown.

Pixels far below the sky, by default more than five noise units, are also
excluded: source detection only looks for positive excursions, and a
negative artefact, such as a bad column, would otherwise enter the fit.
`preparation.masking.negative_threshold_sigma` sets the threshold, and
`null` disables it.

## An external mask

A mask of your own, for example drawn by hand around a satellite trail or
the halo of a bright star, is a FITS image of the same size as the input
image, non-zero where pixels are excluded:

```yaml
data:
  mask_path: data/galaxy_mask.fits
  mask_mode: union     # union, or exact
```

| `mask_mode` | Excluded pixels |
| --- | --- |
| `union` | the automatic mask and the external mask together (default) |
| `exact` | the external mask only, used as it is |

`exact` is the choice when you have already decided every pixel, and
`replace` is an alias of it.

## Looking at the mask

```python
import numpy as np
from baymorph.plotting import plot_log_image

image = runner.prepare_image()

print(f"Excluded pixels: {image.fitting_mask.mean():.1%}")

used = np.where(image.fitting_mask, np.nan, image.prepared_data)
fig, ax = plot_log_image(used, title="Pixels used by the likelihood", show=True)
```

| Attribute | Content |
| --- | --- |
| `fitting_mask` | The mask of the fit: `True` for excluded pixels. |
| `contaminant_mask` | The other detected sources that the contaminant rule masks. |
| `released_labels` | The labels of the detected segments that the rule left unmasked. |
| `target_segment` | The pixels detected as the target. |
| `segmentation` | The segmentation map: 0 for the background, one integer label per source. |
| `external_mask` | Your external mask on the cutout, when one is given. |

!!! tip "Always look at the mask"

    Deblending can split the outskirts of the target itself, such as a
    spiral arm or a bright clump, into separate sources, which are then
    masked as if they were neighbours. If part of your galaxy is masked,
    supply an external mask with `mask_mode: exact`.

The mask is part of the scientific problem: the likelihood is computed only
on the pixels it leaves in, so fits with different masks, and their
evidences, are not directly comparable. A mask chosen after looking at the
residuals of a fit makes the analysis depend on that fit; report it as
such.
