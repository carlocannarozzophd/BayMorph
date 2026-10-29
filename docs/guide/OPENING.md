# Opening an Image

Before BayMorph fits anything, it opens the FITS file and reads what the
header says about the image: which HDU holds it, its units, its calibration
and its pixel scale. This page shows how to see what BayMorph has read, how
to declare what the header does not say, and how BayMorph treats large
mosaics, compressed files and pixels without data.

## The image

BayMorph reads a two-dimensional image in any units: counts, ADU, electrons
per second or calibrated fluxes. The priors, the residual background and the
effective gain are then expressed in the same units. The sky may still be in
the image or may have been subtracted already; BayMorph measures and removes
what is left.

The image is the primary HDU when it holds one. Otherwise BayMorph takes the
single extension named `SCI` or `IMAGE`, as in HST, JWST and HSC products,
or the single image extension, and inherits the keywords of the primary
header. When a file holds several images, such as the detectors of a mosaic
camera, name the one to analyse:

```yaml
data:
  image_path: data/mosaic.fits
  hdu: S4          # extension name, or its index
```

`baymorph config --hdu` writes the same key. A joint fit of several
observations does not accept `data.hdu` yet: each of its images must be the
one BayMorph selects by default.

BayMorph fits pixel values that are linear in flux. It refuses an image
whose header declares another encoding, such as a Pan-STARRS stack, whose
values are compressed through `asinh` (`BSOFTEN` and `BOFFSET`); the FITS
cutouts of the PS1 image cutout service, and the cutouts that astrocut
linearizes, are linear and load as usual.

## What the header says

The observation advisor prints what BayMorph has read from the header:

```python
from baymorph import ObservationAdvisor

advisor = ObservationAdvisor.from_project("data/galaxy.fits")
print(advisor.header_report())
```

The report reads only the header, never the pixels, so the report of a
mosaic of several gigabytes costs the same as the report of a small stamp.
`baymorph config --advise` prints the same report before it prepares the
image.

The report names the conventions that apply to the header and lists, for
each quantity the header holds, its value, its unit, its status and its
source: the keyword, or the keywords of a formula, with the convention that
read it. Notes follow, one per quantity that needs one, such as the formula
of a derived zeropoint or the reason why a value is not used. The status
says how certain a value is:

| Status | Meaning | Used |
| --- | --- | --- |
| `declared` | You declared it in `data.calibration`. | yes |
| `explicit` | The header states it: a text, a count, or a number whose unit a unit keyword or the comment of its card gives. | yes |
| `convention` | The convention that read it, the FITS standard or the documentation of the product, gives its meaning and its unit. | yes |
| `derived` | BayMorph computed it from other keywords, such as an AB zeropoint from `PHOTFLAM` and `PHOTPLAM`, or the pixel scale from the celestial WCS. | yes |
| `ambiguous` | The header does not establish its unit or its meaning, for example a saturation level in ADU in an image in ADU per second. | no |
| `conflicting` | Two keywords, or two conventions, give values that disagree. | no |

A quantity the header does not hold is left out of the report. BayMorph
never infers a unit from a value: an `ambiguous` or `conflicting` quantity is
not used until you declare it.

## Known products

BayMorph reads the header through conventions, one for each family of
products whose documentation says what their keywords mean, and through the
FITS standard otherwise. The conventions cover ESO products and the VST-Tube
pipeline, SWarp coadds, SDSS frames, the coadds of the Legacy Surveys, HST
and JWST images, two High Level Science Products of HST, DECam images of the
NOIRLab Community Pipeline, hscPipe images that record `FLUXMAG0`,
Pan-STARRS stacks, the cutouts that astrocut linearizes, and the mosaics of
grizli, such as those of the DAWN JWST Archive.
[Supported Images](../reference/SUPPORTED_IMAGES.md) says, for each of them,
what BayMorph reads and what you have to declare.

Where a quantity follows from others, BayMorph derives it only for the
products whose documentation covers it. The AB zeropoint of HST follows from
`PHOTFLAM` and `PHOTPLAM` for drizzled images in electrons per second, and
the zeropoint of a resampled JWST image (`i2d`) in MJy/sr follows from its
mean pixel area, `PIXAR_SR`.

When several conventions read the same quantity, the most specific one
decides, and BayMorph compares its value with the others: two values that
disagree make the quantity `conflicting`, with both sources in its note. Two
zeropoints agree when they differ by at most 0.001 mag, which changes a flux
by less than 0.1 percent. A convention whose documentation corrects another
one overrides it instead, and the note records the value it overrode: an HST
image drizzled to counts per second (`D001OUUN = 'cps'`) from exposures in
electrons is read in electrons per second even when its `BUNIT` still says
`ELECTRONS`.

## Declaring what the header lacks

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

| Key | Quantity |
| --- | --- |
| `image_unit` | Unit of the pixel values, such as `ADU/s` or `electron/s`. |
| `zeropoint` | Zeropoint in magnitudes, with m = zeropoint − 2.5 log10(flux in image units). |
| `photometric_system` | `AB` or `Vega`, the system of the declared zeropoint; only together with `zeropoint`. |
| `exposure_time` | Exposure time in seconds. |
| `gain` | Gain of the detector in electrons per ADU, the meaning of the `GAIN` keyword; not the effective gain of the noise model. |
| `saturation` | Saturation level in the units of the image. |

A declared value takes the place of the header reading, with the status
`declared`, and the header reading stays recorded beside it in every report.
A declared unit also reaches the quantities that depend on it: declaring
`electron/s` lets the HST convention derive the zeropoint from `PHOTFLAM`
and `PHOTPLAM` when they are valid. A zeropoint that the header gives is
kept, even when it comes from an earlier calibration: declare one to replace
it. Declarations change no prepared product.

The same declarations go to `baymorph config` as options, `--image-unit`,
`--zeropoint`, `--photometric-system`, `--exposure-time`, `--gain` and
`--saturation`, and to the observation advisor when you create it:

```python
advisor = ObservationAdvisor.from_project(
    "data/galaxy_drz.fits",
    calibration={"zeropoint": 25.94, "photometric_system": "AB"},
)
```

Every reading of the advisor then uses them, and the configuration it
proposes carries them in `data.calibration`. A joint fit does not accept
`data.calibration` yet.

A header without a celestial WCS gives no pixel scale. `data.pixel_scale`
declares it, in arcsec per pixel, for the working region and the reports;
the lengths that the advisor proposes then keep their default ranges in
pixels, and a target cannot be given in RA and Dec.

For a drizzled ACS image whose header gives no usable zeropoint, the
observation advisor asks the ACS Zeropoint Calculator of STScI for it, unless
you declare one or keep the advisor offline, and writes the answer to
`data.calibration`; see
[Writing a Configuration](CONFIGURING.md#interactively-in-a-notebook).

## Large and compressed images

A mosaic can hold far more pixels than a fit needs: 21 504 x 16 384 pixels
take 2.8 GB in double precision. BayMorph reads an image with a side larger
than 4096 pixels only within a working region, a square of 4096 pixels about
the target, about 134 MB in double precision whatever the size of the mosaic.
Set the region with `data.region`:

```yaml
data:
  region: {size: 300}   # a square of 300 arcsec about the target
  # region: whole       # the whole image, whatever its size
```

`baymorph config --region-size` and `--whole-image` write the same key, and
the command says how much of the image BayMorph will read. The region is
shifted inside the image and never padded. Every position, of the target, of
the cutout or of the PSF stars, stays in pixels of the whole image. A cutout
that leaves the region stops with the side to declare. Within a region, the
first detection of an automatic cutout and the search of the PSF stars cover
the region only.

Only the pixels of the region are read from an uncompressed file, and only
the tiles that cross it from a tile-compressed file such as a `.fits.fz`. A
file compressed as a whole, with gzip (`.fits.gz`) or bzip2, is read through
up to the region, in one pass. The interactive selection of the target, in a
notebook, always shows the whole image.

## Pixels without data

Mosaics and drizzled images fill the parts of their grid that no exposure
covers with NaN or with zeros. BayMorph treats as pixels without data the
NaN or infinite pixels and, in an image stored in floating point, the areas
of at least 100 connected pixels that are exactly zero. It leaves them out
of the detection of the sources, of the statistics of the image and of the
stars of the PSF, wherever they are: a star whose window touches them is not
used.

The cutout must hold none of them, unless an external mask excludes them
(see [Masks](MASKS.md)): otherwise the preparation stops and says where they
are, so that a smaller cutout or a mask can leave them out. BayMorph never
guesses a value for them.

## Next

Once the image is open, [Target and Cutout](IMAGE.md) says how BayMorph
finds the galaxy and cuts it out, and
[Writing a Configuration](CONFIGURING.md) how to write the analysis.
