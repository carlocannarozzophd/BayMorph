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
| `--sky-correlation` | Estimator of the sky correlation of the noise model: `autocovariance`, or `field_separated` for a sky that holds a large-scale field; see [Noise Models](../models/NOISE.md). | `autocovariance` |
| `--outputs-root` | Root directory of the run products. | `runs` |
| `--overwrite` | Write `outputs.overwrite: true`, which allows products of existing runs to be replaced. | off |
| `--advise` | Let the observation advisor propose the configuration, as described below. | off |
| `--accept` | With `--advise`, write the reviewed proposal. | off |
| `--background-model` | With `--advise`, the residual background model whose priors are proposed: `constant` or `plane`. | `constant` |
| `--resampling-variance-factor` | With `--advise`, the factor by which a resampling of the image multiplied the variance of its noise, which lets the gain keyword of a resampled image be proposed as the effective gain. | |

The generated file is a starting point to review, not a finished analysis:
[Writing a Configuration](../guide/CONFIGURING.md) lists the choices that are
yours, such as the defaults that are not scaled to the units of the image.
When the sky of the image, measured after three-sigma clipping, is at least
ten times its noise, the file declares `effective_gain` with
`source: measured_sky`; otherwise it leaves the gain undeclared, the command
says so, and the fit refuses to start until `fit.likelihood.effective_gain`
is declared (see [Sky and Noise](../guide/SKY_AND_NOISE.md)).

### With the observation advisor

With `--advise`, the command prints what the header holds, prepares the
image and builds its PSF in the run directory, where the fit reuses them,
and prints a proposal of every prior of every `--model`, with its source:
the header, the prepared sky scale and PSF, the BayMorph defaults, or a
`--uniform-prior` you declared. It writes nothing; run the same command
again with `--accept` to write the YAML and, beside it, a report
`<stem>.advisor.json` of where every proposed value comes from. The rules
of the proposal are those of
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
