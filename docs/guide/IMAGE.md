# Image and Cutout

## The input image

BayMorph reads a two-dimensional image from a FITS file, in any units:
counts, ADU, electrons per second or calibrated fluxes. The priors, the
residual background and the effective gain are then expressed in the same
units. The sky may still be in the image or may have been subtracted
already; BayMorph measures and removes what is left.

The image is the primary HDU when it holds one. Otherwise BayMorph takes
the single extension named `SCI` or `IMAGE`, as in HST, JWST and HSC
products, or the single image extension, and inherits the keywords of the
primary header. When a file holds several images, such as the detectors of
a mosaic camera, name the one to analyse:

```yaml
data:
  image_path: data/mosaic.fits
  hdu: S4          # extension name, or its index
```

A joint fit of several observations does not accept `data.hdu` yet: each
of its images must be the one BayMorph selects by default.

BayMorph reads the header through conventions for the instruments and
pipelines it knows, and through the FITS standard otherwise. Each quantity,
such as the pixel scale, the gain, the zeropoint or the FWHM, carries the
keyword it comes from and how certain it is, and BayMorph never infers a
unit from a value. The conventions cover ESO products and the VST-Tube
pipeline, SWarp coadds, SDSS frames, the coadds of the Legacy Surveys, HST
and JWST images, DECam images of the NOIRLab Community Pipeline, hscPipe
images that record `FLUXMAG0`, and Pan-STARRS stacks. Where a quantity
follows from others, such as the AB zeropoint of HST from `PHOTFLAM` and
`PHOTPLAM`, BayMorph derives it only for the products its documentation
covers, here the drizzled images in electrons per second.

When the header lacks a calibration quantity, or states it wrongly, declare
it in `data.calibration`. A drizzled image whose header says `ELECTRONS`
while its pixels are electrons per second, and whose `PHOTFLAM` is 0, takes:

```yaml
data:
  image_path: data/galaxy_drz.fits
  calibration:
    image_unit: electron/s
    zeropoint: 25.94        # from the documentation of the product
    photometric_system: AB
```

A declared value takes the place of the header reading, with the status
`declared`, and the header reading stays recorded beside it in every report.
A declared unit also reaches the quantities that depend on it, so declaring
`electron/s` lets the HST convention derive the zeropoint from `PHOTFLAM` and
`PHOTPLAM` when they are valid. The quantities that may be declared are
listed in the [configuration reference](../reference/CONFIGURATION.md).
A joint fit does not accept `data.calibration` yet.

BayMorph fits pixel values that are linear in flux. It refuses an image
whose header declares another encoding, such as a Pan-STARRS stack, whose
values are compressed through `asinh` (`BSOFTEN` and `BOFFSET`); the FITS
cutouts of the PS1 image cutout service, and the cutouts that astrocut
linearizes, are linear and load as usual.

Every pixel must be finite. BayMorph refuses an image with NaN or infinite
pixels instead of guessing a value for them: fill them before, and list them
in an external mask so that the fit ignores them (see [Masks](MASKS.md)).

## The target

`data.target_position` tells BayMorph which source to fit. It is a position
in zero-based pixel coordinates of the full image, `x` along the columns
and `y` along the rows, so that the pixel is `image[y, x]`; `center` uses
the centre of the image. Any point on the galaxy will do: BayMorph detects
the sources in the image, takes the one that contains this position as the
target, and fits its centre as a free parameter.

## The cutout

The fit does not see the whole image, only a square cutout around the
target. The cutout must contain the whole galaxy and enough empty sky
around it, where BayMorph measures the sky and the noise.

=== "Fixed"

    ```yaml
    data:
      cutout:
        mode: fixed
        size: 251        # pixels
    ```

    A square of `size` pixels centred on the target position. Choose it
    from the extent of the galaxy: the light of the galaxy must reach the
    sky well inside the cutout.

=== "Automatic"

    ```yaml
    data:
      cutout:
        mode: auto
        sky_margin: 20   # pixels added beyond the edge of the galaxy
    ```

    BayMorph cuts a large preliminary region, measures the radius where the
    radial profile of the galaxy flattens into the sky, the plateau, and
    cuts the plateau radius plus `sky_margin` pixels on each side.
    `min_size` and `max_size` bound the result. When the cutout cannot
    contain the plateau and its margin, preparation stops with an error
    rather than fitting a galaxy that does not fit in its cutout; a fixed
    cutout is then the remedy.

The cutout, together with the mask, is the part of the image the likelihood
uses. A different cutout is a different scientific problem: results from
two cutouts are not directly comparable, and neither are their evidences.

## Preparing the image

```python
from baymorph import BayMorphRunner

runner = BayMorphRunner("configs/galaxy.yaml")
image = runner.prepare_image()
```

`prepare_image()` cuts out the target, detects and masks the other sources,
subtracts the sky and measures the noise. It returns the prepared
observation:

```python
image.prepared_data.shape    # (251, 251): the cutout, sky subtracted
image.cutout_mode            # 'fixed'
image.cutout_bounds          # {'xmin': 375, 'xmax': 626, 'ymin': 375, 'ymax': 626}
image.target_x, image.target_y   # (125, 125): the target in the cutout
```

| Attribute | Content |
| --- | --- |
| `prepared_data` | The cutout with the sky subtracted: the data of the fit. |
| `cutout` | The same cutout before sky subtraction. |
| `cutout_bounds` | Where the cutout lies in the full image. |
| `fitting_mask` | `True` where a pixel is excluded from the fit; see [Masks](MASKS.md). |
| `noise` | The noise scale of the sky in each pixel; see [Sky and Noise](SKY_AND_NOISE.md). |
| `background` | The sky level that was subtracted. |
| `segmentation`, `target_segment`, `contaminant_mask` | The detected sources, the target, and the other sources that the contaminant rule masks. |
| `sky_gradient` | The gradient of the sky across the cutout, a diagnostic. |

Positions in the results, such as the fitted centre `x0`, `y0`, are in
pixels of the cutout. Add `cutout_bounds["xmin"]` and
`cutout_bounds["ymin"]` to place them in the full image.

To look at the prepared cutout:

```python
from baymorph.plotting import plot_log_image

fig, ax = plot_log_image(image.prepared_data, title="Prepared cutout", show=True)
```

The products are also written to disk, under `image_preparation/` in the
run directory; a later run of the same configuration reuses them.
