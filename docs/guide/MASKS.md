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
levels. The source that contains the target position is the target; every
other source is masked.

```yaml
preparation:
  detection:
    n_sigma: 3.0       # detection threshold, in noise units
    n_pixels: 20       # smallest source, in connected pixels
    n_levels: 32       # deblending levels
```

Each source is masked over its detected segment. The faint outskirts of a
bright neighbour beyond its segment are not masked; an external mask can
cover them.

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
| `contaminant_mask` | The other detected sources. |
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
