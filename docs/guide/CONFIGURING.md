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

`target_position` is the position of the target in zero-based pixel
coordinates of the full image: `x` is the column and `y` the row, as in
`image[y, x]`. It only tells BayMorph which source to fit; the centre of the
galaxy is a fitted parameter.

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

It prints what the header holds, prepares the image and builds its PSF,
and prints the proposal with the source of every prior. Nothing is written
until you run the same command again with `--accept`. Several `--model`
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
advisor.select_target()           # click on the galaxy, then confirm
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
[The image](IMAGE.md)); its status then becomes `declared`.

The Advisor first prepares the image and builds its PSF, the inputs of the
observation model that come before any prior; the fit then reuses them. To
look at them before the priors, call `advisor.measure_fixed_inputs(...)` and
pass the result to `propose_config(fixed_inputs=...)` with the same options,
as the tutorial does. The Advisor then proposes the priors from the header,
from physics and from those inputs, never from the image it is about to
fit:

| Parameter | Proposed prior |
| --- | --- |
| Fluxes | Magnitudes 8 to 30, converted through the zeropoint of the header: log-uniform in image units. |
| Lengths of a model with one component | 0.1 to 120 arcsec, converted through the pixel scale: log-uniform in pixels, within the range where the rendering of the model is validated. |
| Lengths of a model with several components | The default ranges, which tell the components apart, limited to 0.1 to 120 arcsec. |
| Centre | Within three times the FWHM of the PSF of the target position. |
| Sérsic index, axis ratio, angles | The BayMorph defaults. |
| Residual background | Within five times the sky scale; a plane changes by at most as much over half the cutout. |

It also proposes where the effective gain comes from: the sky level the
header records, the sky measured in the image, or the gain keyword, in that
order, and it compares them when it finds more than one. When the header
gives the saturation level in the units of the image, half of it becomes
the brightest star allowed for the PSF.

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
proposal of one model with one total flux. Without a zeropoint in the
header, the flux priors must be declared in image units; without a pixel
scale, the lengths keep the default ranges in pixels, and the summary says
so.

Without the interactive widgets, `advisor.set_target_position(x, y)` sets
the target instead of the click. Nothing is written until
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
| The cutout | `data.cutout` | The cutout is the part of the image the fit sees; see [Image and Cutout](IMAGE.md). |

The Advisor proposes each of these and records where each value comes
from, but accepting them stays your decision.

!!! tip "Priors are assumptions"

    A prior states what you know before looking at the image. Choose it
    from the physics of the object or the survey, for example a magnitude
    range converted to image units, and not by looking at the fitted
    values: a prior tuned on the result makes the result circular.
