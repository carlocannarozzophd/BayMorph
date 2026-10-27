# Writing a Configuration

A BayMorph analysis is described by one YAML file. It names the image and
the target, and declares the model, the priors, the noise model and the
sampler. BayMorph fills in every option you leave out with its default and
writes the complete, resolved configuration next to the results, so every
analysis can be repeated exactly.

## A minimal configuration

```yaml
object:
  name: galaxy                  # names the output directory

data:
  image_path: data/galaxy.fits
  target_position: {x: 500, y: 500}
  cutout:
    mode: fixed
    size: 251

psf:
  type: empirical               # built from the stars of the image

fit:
  models: [single_sersic]
  priors:
    single_sersic:
      flux: {type: uniform, lower: 100, upper: 1000000}
  likelihood:
    type: correlated_gaussian
    effective_gain:
      source: measured_sky
  sampler:
    name: nautilus

outputs:
  root: runs
```

`target_position` is the centre of the galaxy in zero-based pixel
coordinates of the full image: `x` is the column and `y` the row, as in
`image[y, x]`. It selects the source to fit, and the fit samples the centre
of the galaxy within a window about it, so it must be the centre (see
[Target and Cutout](IMAGE.md#the-target)). `baymorph config --target-ra RA
--target-dec DEC`, and the advisor, take the position on the sky instead and
write it in pixels.

Every key, with its default, is listed in the
[Configuration reference](../reference/CONFIGURATION.md).

## Three ways to write it

### From the terminal

```bash
baymorph config data/galaxy.fits \
  --target-x 500 --target-y 500 \
  --model single_sersic \
  --uniform-prior flux 100 1000000 \
  --output configs/galaxy.yaml
```

`--uniform-prior` can be repeated for any parameter. The options are listed
in [Command Line](../reference/CLI.md).

With `--advise`, the command lets the observation advisor described below
propose the configuration instead:

```bash
baymorph config data/galaxy.fits \
  --target-x 500 --target-y 500 \
  --advise \
  --model single_sersic --model double_sersic \
  --output configs/galaxy.yaml
```

It prints what the header holds, prepares the image and builds its PSF in
the run directory, where the fit reuses them, and prints the proposal with
the source of every prior. The configuration is written only when you run
the same command again with `--accept`. Several `--model`
options give one configuration with every model, each fitted separately;
with several models, a declared prior names its model, as in
`--uniform-prior double_sersic.q_bulge 0.4 1`.

### From Python

```python
from baymorph import make_config_from_fits

config = make_config_from_fits(
    "data/galaxy.fits",
    target_position={"x": 500, "y": 500},
    model_name="single_sersic",
    model_priors={"flux": {"type": "uniform", "lower": 1e2, "upper": 1e6}},
    background={
        "model": "constant",
        "priors": {
            "background_level": {"type": "uniform", "lower": -20, "upper": 20},
        },
    },
    psf_max_peak=50000,
    cutout_mode="fixed",
    fixed_size=251,
    output_path="configs/galaxy.yaml",
)
```

The function returns the configuration as a dictionary and writes it when
`output_path` is given.

### Interactively, in a notebook

`ObservationAdvisor` reads the image and its header, lets you click on the
target, proposes a complete configuration and writes the YAML only after
you have reviewed and accepted it. The tutorial notebook
`02A_Observed_Assessment_and_Config.ipynb` walks through it:

```python
from baymorph import ObservationAdvisor

advisor = ObservationAdvisor.from_project("data/galaxy.fits")
print(advisor.header_report())    # what the header holds
advisor.select_target()           # click on its centre, then confirm
advisor.inspect()
print(advisor.summary())          # diagnostics of the image and the target

proposal = advisor.propose_config(
    output_path="configs/galaxy.yaml",
    models=["single_sersic", "double_sersic"],
    background_model="plane",
)
print(proposal.summary())
proposal.write(accept=True)
```

`header_report()` lists each quantity read from the header with its value,
unit, status and source. A quantity whose unit or meaning the header does
not establish is marked `ambiguous` and is never used: a unit is never
guessed from a value. Declare such a quantity in `data.calibration` (see
[Opening an Image](OPENING.md#declaring-what-the-header-lacks)); its status
then becomes `declared`. Give the declarations to the Advisor when you
create it, for example
`ObservationAdvisor.from_project(fits_path, calibration={"zeropoint": 25.94,
"photometric_system": "AB"})`: every reading of the Advisor then uses them,
and the proposed configuration carries them in `data.calibration`. On the
command line the options `--image-unit`, `--zeropoint`,
`--photometric-system`, `--exposure-time`, `--gain` and `--saturation` do the
same (see [Command Line](../reference/CLI.md)).

Some calibration quantities are given only by an online service of the
observatory. When the header leaves such a quantity ambiguous or missing and
you declare none, the Advisor asks the service once, before it prepares the
image, and writes the answer to `data.calibration`, so that the fit reads it
from the YAML and never goes online. A value you declare, or one the header
gives, is never replaced, and two readings of the header that disagree are
never settled by a service.

| Service | Images | What the Advisor sends | What it writes |
| --- | --- | --- | --- |
| [ACS Zeropoint Calculator](https://acszeropoints.stsci.edu) of STScI | ACS images of the WFC or HRC detector drizzled to electrons per second (`D001OUUN = 'cps'`), observed within one year (`EXPSTART`, `EXPEND`) | the detector, the filter, and the date of the middle of the observation; for an observation over several dates, also its first and last dates | `zeropoint`, the AB zeropoint of the middle date, and `photometric_system: AB` |

The sensitivity of ACS does not only drift: for WFC F814W the calculator
gives 25.948 on 2006-06-20 and 25.932 on 2006-07-20. When the zeropoints of
the first and last dates of an observation differ from that of its middle
date by more than 0.002 mag, no single zeropoint holds for its exposures,
and the Advisor writes none.

The header report and the summary say what was asked and answered, when
an answer was applied, or why the service was not asked, for example for a mosaic without
observation dates; the report beside the YAML records the request and the
answer. The answer is accepted only when it is consistent with itself: its
AB and ST zeropoints must agree with its `PHOTFLAM` and pivot wavelength.
Without a usable answer, the fluxes take the priors that need no zeropoint.
Proposals of one image made with and without the answer therefore have
different flux priors, and their evidences are not comparable.
`ObservationAdvisor(..., online=False)` asks nothing, as for a batch of
images on a computer without network. Writing the same proposal again keeps
the report that is already there when it differs only in when the service
was asked, or in the verbatim text of the error a failed request met;
another answer, after a recalibration of the observatory, even one that
fails for the same reason, or a failure for another reason, is a different
proposal, and `write` refuses to replace the files and names the keys that
differ.

The Advisor first prepares the image and builds its PSF, the inputs of the
observation model that come before any prior; the fit then reuses them. To
look at them before the priors, call `advisor.measure_fixed_inputs(...)` and
pass the result to `propose_config(fixed_inputs=...)` with the same options,
as the tutorial does. The Advisor then proposes the priors from the header,
from physics and from those inputs, never from the image it is about to
fit:

| Parameter | Proposed prior |
| --- | --- |
| Fluxes | Magnitudes 8 to 30, converted through the zeropoint, read from the header, declared, or given by a calibration service: log-uniform in image units. Without a zeropoint, from the flux of a point source at signal-to-noise 1 on the prepared sky, the sky scale times the square root of the noise-equivalent area of the PSF, to 22 magnitudes above it: log-uniform in image units. |
| Lengths of a model with one component | 0.1 to 120 arcsec, converted through the pixel scale: log-uniform in pixels, within the range where the rendering of the model is validated. |
| Lengths of a model with several components | The default ranges, which tell the components apart, limited to 0.1 to 120 arcsec. |
| Centre | Within three times the FWHM of the PSF of the target position. |
| Sérsic index, axis ratio, angles | The BayMorph defaults. |
| Residual background | Within five times the sky scale; a plane changes by at most as much over half the cutout. |

It also proposes where the effective gain comes from: the sky level the
header records, the sky measured in the image, or the gain of the detector,
in that order, and it compares them when it finds more than one. The gain of
the detector is the gain keyword for an image in ADU, 1 for an image in
electrons, and the exposure time for an image in electrons per second. For a
resampled image, such as a drizzled one, and for any image in electrons per
second, it is proposed only when you declare the resampling variance factor,
1 for a single exposure that was not resampled, and is divided by it. When
the header gives the saturation level in the units of the image, half of
it becomes the brightest star allowed for the PSF.

`models` holds one or more models, written to `fit.models`. Each receives
its priors by the same rules, and the fit runs one fit per model on the same
prepared image, PSF and likelihood, each with its own posterior and
evidence. The choice of the models is yours: BayMorph distributes no
validated classifier that ranks them, and fitting several is not a
preference for any of them.

Anything you declare is kept as declared: a prior through `model_priors`
or `flux_prior`, `background`, `likelihood` or `psf_max_peak`. With
`models`, `model_priors` maps each model to its declared priors, for
example `{"double_sersic": {"q_bulge": {...}}}`; `flux_prior` applies to a
proposal of one model with one total flux. Each proposed flux prior states
its rule and its numbers beside it, with or without a zeropoint; without a
pixel scale, the lengths keep the default ranges in pixels, and the summary
says so.

Without the interactive widgets, `advisor.set_target_position(x, y)` sets
the target instead of the click. The configuration is written only by
`proposal.write(accept=True)`. The summary lists every prior with its
source, and the written YAML comes with a report, `galaxy.advisor.json`,
that records where every proposed value comes from. The diagnostics the
Advisor prints, such as a flux or radius measured on the image, are there
to check that a prior is sensible, never to set its bounds.

## What you have to decide

A configuration generator writes a file you can run, but some choices are
yours, because BayMorph never derives them from the image it is about to
fit:

| Choice | Where | Why it matters |
| --- | --- | --- |
| The flux prior | `fit.priors.<model>.flux` | No model has a default flux prior: the range depends on the units and depth of your image. |
| The other priors | `fit.priors.<model>` | The defaults are wide uniform ranges in pixels; check them against the size of your galaxy. |
| The residual background prior | `fit.background.priors` | The default, −50 to 50 in image units, is not scaled to your image. |
| The effective gain | `fit.likelihood.effective_gain` | It sets the Poisson noise of the galaxy; see [Sky and Noise](SKY_AND_NOISE.md). |
| The brightest star for the PSF | `psf.empirical.candidate_selection.max_peak` | The default, 80000, is not scaled to your image: keep it below the saturation level, so that saturated stars are not used. |
| The cutout | `data.cutout` | The cutout is the part of the image the fit sees; see [Target and Cutout](IMAGE.md). |
| What the header does not say | `data.calibration` | A quantity that the header leaves `ambiguous` or `conflicting` is not used until you declare it; see [Opening an Image](OPENING.md). |

The Advisor proposes each of these and records where each value comes
from, but accepting them stays your decision.

!!! tip "Priors are assumptions"

    A prior states what you know before looking at the image. Choose it
    from the physics of the object or the survey, for example a magnitude
    range converted to image units, and not by looking at the fitted
    values: a prior tuned on the result makes the result circular.
