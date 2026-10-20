# Supported Images

BayMorph opens any two-dimensional FITS image whose pixel values are linear
in flux. For the families of products on this page, a convention written
from their documentation tells BayMorph what their keywords mean; for any
other image, BayMorph reads the keywords of the FITS standard. Whatever the
header does not establish is never used, and you declare it in
`data.calibration`. [Opening an Image](../guide/OPENING.md) shows how to see
what BayMorph has read, and how to declare the rest.

## What BayMorph uses from a header

| Quantity | Used for | Without a usable value |
| --- | --- | --- |
| Pixel scale, from the celestial WCS | the lengths that the advisor proposes, 0.1 to 120 arcsec; a working region in arcsec; a target in RA and Dec | the proposed lengths keep their default ranges in pixels; declare `data.pixel_scale` for a region in arcsec |
| Zeropoint | the flux priors that the advisor proposes, magnitudes 8 to 30 | the flux priors that need no zeropoint, from the sky and the PSF |
| Image unit | whether a zeropoint can be derived, and whether a value in the units of the image, such as the saturation level, can be used | those quantities are not used |
| Saturation level, in the units of the image | half of it is the brightest star the advisor accepts for the PSF | the default ceiling, 80000, unless you set `--psf-max-peak` |
| Sky level subtracted before BayMorph, gain | where the effective gain of the noise model may come from; see [Sky and Noise](../guide/SKY_AND_NOISE.md#the-effective-gain) | the effective gain comes from the sky of the image, or you declare it |
| FWHM of the PSF | a Gaussian PSF requested without its FWHM | give `--gaussian-fwhm-pixels` |
| Detector, filter, dates of the observation | a request to a calibration service | no request |

The exposure time, the telescope, the instrument and the other quantities
are recorded in the reports.

## Every image

The FITS standard and the common keyword dictionaries apply to every header,
and the conventions below take precedence where they read the same quantity.

| Quantity | Keyword | Read as |
| --- | --- | --- |
| Telescope, instrument, filter | `TELESCOP`, `INSTRUME`, `FILTER` | texts |
| Image unit | `BUNIT` | the unit of the pixel values |
| Exposure time | `EXPTIME` | seconds |
| Total exposure time | `TEXPTIME` | seconds, when the comment of the card says so |
| Gain | `GAIN` | electrons per ADU |
| Saturation level | `SATURATE`, or its synonym `SATLEVEL` | the units of the image |
| FWHM of the PSF | `FWHM` | arcsec or pixels, as `FWHMUNIT`, `SEEUNIT` or the comment of the card states |
| Zeropoint | `PHOTZP` | `ambiguous`: outside the conventions that define it, its meaning is not established |

The pixel scale comes from the celestial WCS, as the mean of its projected
scales along the two axes.

## Product families

Each family is recognized by a keyword of its header. A product that no
family recognizes is read through the keywords of every image.

### Ground-based surveys and pipelines

| Products | Recognized by | BayMorph reads | You declare |
| --- | --- | --- | --- |
| ESO science data products (Phase 3) | `PRODCATG` | the zeropoint `PHOTZP`, with mag = −2.5 log10(data) + `PHOTZP`, its error `PHOTZPER` and its system `PHOTSYS`; `EXPTIME`, the total integration time per pixel; `TEXPTIME`; `NCOMBINE`; `ABMAGLIM`; the FWHM `PSF_FWHM` in arcsec | |
| ESO raw frames | a keyword that starts with `ESO DET` | the read noise of the first output, in electrons, and the filter | the gain, when you need it: the ESO detector manual warns that its two keywords have been confused, so the direction of the conversion is not established |
| Coadds of the VST-Tube pipeline | `PROCSOFT` starting with `VST-tube` | the zeropoint `PHOTZP`, with mag = −2.5 log10(ADU) + `PHOTZP` as its comment defines it, its error and its system; `NCOMBINE`; `TEXPTIME`; the subtracted sky level `BKGMED`, in the units of the coadd; the astrometric error | the saturation level, when you need it: its comment gives ADU, while a coadd may be in ADU per second; the FWHM, whose unit no document of the pipeline states |
| Coadds written by SWarp | `SOFTNAME = 'SWarp'` | the equivalent gain of the coadd, `GAIN`, and its saturation level, `SATURATE` | |
| SDSS corrected frames | `NMGY` | a zeropoint of 22.5 for pixels in nanomaggies, on the AB system, with the percent-level offsets of the SDSS calibration | the gain, when you need it: SDSS tabulates it by camera column and filter instead of writing it in the header |
| Coadds of the Legacy Surveys | `LEGPIPEV` | a zeropoint of 22.5 for pixels in nanomaggies, without a photometric system: the optical bands are AB and the WISE bands Vega | |
| DECam images of the NOIRLab Community Pipeline | `INSTRUME = 'DECam'` | the zeropoint `MAGZERO` for pixel values in ADU | the detector, with `data.hdu`, in a file with one extension per detector; the zeropoint of an image in ADU per second, for which the pipeline writes `MAGZPT`; the saturation level, when you need it: `SATURATA` and `SATURATB` give the two amplifiers in ADU before the calibration of the image |
| Images of hscPipe that record `FLUXMAG0` | `FLUXMAG0` | the zeropoint 2.5 log10(`FLUXMAG0`) | |
| Pan-STARRS1 stacks | `BSOFTEN` with `BOFFSET` | the encoding of the pixels, compressed through `asinh`, which BayMorph refuses | use the FITS cutouts of the PS1 image cutout service, which are linear |
| Cutouts linearized by astrocut | the `HISTORY` card of the conversion to linear flux | linear pixels, although `BSOFTEN` and `BOFFSET` remain in the header | |

From hscPipe 7 on, the header of a coadd records no `FLUXMAG0`: the manual
gives 27.0 mag for pixels in ADU, off by a few percent through the aperture
correction, and the zeropoint must be declared.

### Space telescopes

| Products | Recognized by | BayMorph reads | You declare |
| --- | --- | --- | --- |
| HST images | `TELESCOP = 'HST'` | the detector `DETECTOR`; the start and end of the observation, `EXPSTART` and `EXPEND`, as Modified Julian Dates; the filter from `FILTER`, or for ACS from its two filter wheels `FILTER1` and `FILTER2`, leaving out a `CLEAR` element; the AB zeropoint −2.5 log10(`PHOTFLAM`) − 5 log10(`PHOTPLAM`) − 2.408 of a drizzled product (`FILENAME` with `_dr`) in electrons per second | the zeropoint of any other product: `flt` and `flc` exposures need the pixel area map first |
| HST images drizzled to count rates | `D001OUUN = 'cps'` | the unit, electrons per second, when `BUNIT` starts with `ELECTRONS`, whatever `BUNIT` says of the output; the AB zeropoint from `PHOTFLAM` and `PHOTPLAM`, also when `FILENAME` names an input exposure | the zeropoint when `PHOTFLAM` is 0, which means that the pipeline did not compute it; the unit when `BUNIT` does not name electrons, since the counts may then be DN |
| Hubble eXtreme Deep Field mosaics | `XDFVER` | the AB zeropoint of the release for the filter, for pixels in electrons per second: F435W 25.68, F606W 26.51, F775W 25.69, F814W 25.94, F850LP 24.87, F105W 26.27, F125W 26.23, F140W 26.45, F160W 25.94; the release does not correct the drift of the ACS zeropoints, of order one percent, over the years it combines | |
| HST/ACS Coma Cluster Treasury Survey images | `FILENAME` with `_ivm_drz_cl`, from program 10861 | the AB zeropoint of the survey for pixels in electrons per second, F475W 26.068 and F814W 25.937, in place of the zeropoint of `PHOTFLAM`, which some of these headers give as 0 | |
| JWST images | `TELESCOP = 'JWST'` | the AB zeropoint −2.5 log10(10^6 `PIXAR_SR` / 3631) of a resampled image (`FILENAME` with `_i2d`) in MJy/sr; not of a `cal` image, whose pixel area varies across the detector | |
| Mosaics of grizli, such as those of the DAWN JWST Archive | `GRIZLIV` | the AB zeropoint −2.5 log10(`PHOTFNU` / 3631), 28.9 for pixels of 10 nJy; `ZP`, `OPHOTFNU` and `PIXAR_SR` describe the original unit and the native pixel of the inputs, and are not used | |

For a drizzled ACS image of the WFC or HRC detector in electrons per second
whose header gives no usable zeropoint, and for which you declare none, the
observation advisor asks the ACS Zeropoint Calculator of STScI for the
zeropoint of its detector, filter and dates; see
[Writing a Configuration](../guide/CONFIGURING.md#interactively-in-a-notebook).

## When conventions disagree

When two conventions read the same quantity, the more specific one decides,
a product before its instrument and an instrument before the FITS standard,
and BayMorph compares its value with the others. Two values that disagree
make the quantity `conflicting` until you declare it; two zeropoints agree
when they differ by at most 0.001 mag. A convention whose documentation
corrects another overrides it instead, and the header report records the
value it overrode: the drizzle rule overrides `BUNIT`, the zeropoints of the
XDF and of the Coma survey override the one derived from `PHOTFLAM`, and
astrocut overrides the encoding of the Pan-STARRS stack it linearized.
