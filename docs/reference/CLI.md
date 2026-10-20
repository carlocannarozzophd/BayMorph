# Command-Line Interface

The `baymorph` command has four subcommands. `baymorph --version` prints the
installed version and `baymorph <subcommand> --help` lists the options below.

## `baymorph init`

```bash
baymorph init [--force] project_name
```

Creates a project directory with `configs/`, `notebooks/`, `data/` and
`runs/`, and copies the packaged example configuration and tutorial
notebooks into it.

| Option | Meaning |
| --- | --- |
| `project_name` | Directory to create. |
| `--force` | Allow writing into an existing non-empty directory. |

## `baymorph examples`

```bash
baymorph examples [--force] [destination]
```

Copies only the packaged example configurations.

| Option | Meaning |
| --- | --- |
| `destination` | Destination directory; default `./baymorph_examples`. |
| `--force` | Overwrite existing example files. |

## `baymorph config`

```bash
baymorph config fits_path (--target-center | --target-x X --target-y Y) [options]
```

Writes a single-observation configuration for one FITS image. The target
seed is required: either `--target-center`, or `--target-x` and `--target-y`
together.

| Option | Meaning | Default |
| --- | --- | --- |
| `fits_path` | Input FITS image. | required |
| `--target-x`, `--target-y` | Target-identification seed in zero-based full-image pixel coordinates. | |
| `--target-center` | Use the detector centre as the target seed; cannot be combined with `--target-x` or `--target-y`. | |
| `--hdu` | Index or `EXTNAME` of the HDU that holds the image, written to `data.hdu`. | selected by BayMorph |
| `--region-size` | Side, in arcsec, of the square about the target within which the image is read, written to `data.region`. | the whole image up to 4096 pixels on a side, otherwise a square of 4096 pixels about the target |
| `--whole-image` | Read the whole image, whatever its size: `data.region: whole`. Cannot be combined with `--region-size`. | |
| `--output` | Path of the YAML file to write. | `<FITS stem>.yaml` in the current directory |
| `--object-name` | Name of the analysis object, which names its output directory. | FITS file name without extension |
| `--model` | Registered morphology model, one of the ten of the [Models](../models/index.md) section, such as `single_sersic`, `bulge_disk` or `core_sersic`. Repeat it with `--advise` for a configuration of several models. | `single_sersic` |
| `--sampler` | Inference backend: `nautilus`, `dynesty` or `pocomc`. | `nautilus` |
| `--uniform-prior PARAMETER LOWER UPPER` | Uniform prior of one model parameter; repeat it for several. Without `--advise` it is required for every parameter without a default prior, such as `flux`, or `flux_bulge` and `flux_disk`. With several `--model` options, write the parameter as `MODEL.PARAMETER`, such as `double_sersic.q_bulge`. | |
| `--cutout-mode` | `auto` derives the cutout from the sky plateau of the target; `fixed` uses `--cutout-size`. | `auto` |
| `--cutout-size` | Cutout size in pixels for `--cutout-mode fixed`. | |
| `--psf` | PSF strategy: `empirical` or `gaussian`. | `empirical` |
| `--gaussian-fwhm-pixels` | FWHM of a Gaussian PSF in pixels, needed only when the FITS metadata do not provide it. | |
| `--psf-max-peak` | Brightest peak pixel accepted for a star of the empirical PSF, in image units; keep it below the saturation level. | BayMorph default; with `--advise`, half the saturation level of the header when its unit is known |
| `--image-unit` | Unit of the pixel values, such as `ADU/s` or `electron/s`, declared in `data.calibration`. | read from the header |
| `--zeropoint` | Zeropoint in magnitudes, with m = zeropoint − 2.5 log10(flux in image units), declared in `data.calibration`. | read from the header |
| `--photometric-system` | `AB` or `Vega`, the system of the declared zeropoint; only together with `--zeropoint`. | |
| `--exposure-time` | Exposure time in seconds, declared in `data.calibration`. | read from the header |
| `--gain` | Gain of the detector in electrons per ADU, the meaning of the `GAIN` keyword, declared in `data.calibration`; not the effective gain of the noise model. | read from the header |
| `--saturation` | Saturation level in the units of the image, declared in `data.calibration`. | read from the header |
| `--sky-correlation` | Estimator of the sky correlation of the noise model: `autocovariance`, or `field_separated` for a sky that holds a large-scale field; see [Noise Models](../models/NOISE.md). | `autocovariance` |
| `--outputs-root` | Root directory of the run products. | `runs` |
| `--overwrite` | Write `outputs.overwrite: true`, which allows products of existing runs to be replaced. | off |
| `--advise` | Let the observation advisor propose the configuration, as described below. | off |
| `--accept` | With `--advise`, write the reviewed proposal. | off |
| `--offline` | With `--advise`, ask no calibration service online. Without it, the advisor asks the ACS Zeropoint Calculator of STScI for the zeropoint of a drizzled ACS image whose header gives no usable one, unless two readings of it disagree, and for which none is declared. | online |
| `--background-model` | With `--advise`, the residual background model whose priors are proposed: `constant` or `plane`. | `constant` |
| `--resampling-variance-factor` | With `--advise`, the factor by which a resampling of the image multiplied the variance of its noise, which lets the gain keyword of a resampled image be proposed as the effective gain. | |

The generated file is a starting point to review, not a finished analysis:
[Writing a Configuration](../guide/CONFIGURING.md) lists the choices that are
yours, such as the defaults that are not scaled to the units of the image.
The command says how much of the image BayMorph will read: the whole image,
or a working region about the target, with its bounds in pixels of the whole
image. When the sky of the image, measured after three-sigma clipping in the
part that is read, is at least
ten times its noise, the file declares `effective_gain` with
`source: measured_sky`; otherwise it leaves the gain undeclared, the command
says so, and the fit refuses to start until `fit.likelihood.effective_gain`
is declared (see [Sky and Noise](../guide/SKY_AND_NOISE.md)).

### With the observation advisor

With `--advise`, the command prints what the header holds, with the
calibration you declared in place of the header readings, prepares the
image and builds its PSF in the run directory, where the fit reuses them,
and prints a proposal of every prior of every `--model`, with its source:
the header, a declared calibration, the prepared sky scale and PSF, the
BayMorph defaults, or a `--uniform-prior` you declared. When the header of a
drizzled ACS image gives no usable zeropoint, and not two readings of it that
disagree, and you declare none, the advisor first asks the ACS Zeropoint
Calculator of STScI for it, when the image allows a request, sending only the detector, the filter and the
dates of the observation, and prints the answer in a section of its own;
`--offline` asks nothing. Without a zeropoint, read, declared or given by the
calculator, the flux priors come from the sky and the PSF, and the proposal
states the rule and its numbers. It writes nothing; run the same command
again with `--accept` to write the YAML and, beside it, a report
`<stem>.advisor.json` of where every proposed value comes from. With
`--accept` the command writes neither the YAML nor its report, and exits with
status 1, when a request to the calculator failed: no answer, or an answer
that cannot be read or is not consistent with itself. It has then prepared
the image as usual; run it again, or add `--offline` to write the proposal
without the zeropoint. When the image allows no request, or the answers hold
no single zeropoint for an observation over several dates, the proposal is
written with the flux priors that need no zeropoint, and the summary says
why. The rules of the proposal are those of
[Writing a Configuration](../guide/CONFIGURING.md#interactively-in-a-notebook).

The image path and the output root of the written file are absolute, so
the fit finds the products of the advisor wherever it is run from.

## `baymorph run`

```bash
baymorph run [--no-diagnostics] config_path
```

Executes the configured workflow, as `BayMorphRunner(config_path).run()`
does: simulation when the configuration has a `simulation` section, image
preparation, empirical-PSF construction when the PSF is empirical, the
measurement of the sky correlation for the noise model, fitting and
diagnostics. See [The Runner](../guide/RUNNER.md).

| Option | Meaning |
| --- | --- |
| `config_path` | YAML configuration to execute. |
| `--no-diagnostics` | Fit without generating the configured diagnostics. |

## Exit status

| Status | Meaning |
| --- | --- |
| `0` | The command succeeded. |
| `1` | The command failed while it ran, for example because a file is missing or is not a FITS file, a configuration is invalid, or a request to a calibration service that a proposal to write needed failed. The error is printed on one line, `baymorph: error: …`. |
| `2` | The options are invalid or do not go together, for example `--target-center` with `--target-x`. The error is printed after the usage of the command. |
