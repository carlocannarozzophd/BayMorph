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
target, shows prior guidance for each parameter and writes the YAML only
after you have reviewed and accepted it. The tutorial notebook
`02A_Observed_Assessment_and_Config.ipynb` walks through it:

```python
from baymorph import ObservationAdvisor

advisor = ObservationAdvisor.from_project("data/galaxy.fits")
advisor.select_target()           # click on the galaxy, then confirm
advisor.inspect()
print(advisor.summary())

proposal = advisor.propose_config(
    output_path="configs/galaxy.yaml",
    model_name="single_sersic",
    flux_prior={"type": "log_uniform", "lower": 1e3, "upper": 1e7},
    background={
        "model": "constant",
        "priors": {
            "background_level": {"type": "uniform", "lower": -20, "upper": 20},
        },
    },
    psf_max_peak=50000,
)
print(proposal.summary())
proposal.write(accept=True)
```

Without the interactive widgets, `advisor.set_target_position(x, y)` sets
the target instead of the click. Nothing is written until
`proposal.write(accept=True)`, and the summary lists the priors left at
their defaults, so that you can review them first. The diagnostics the
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

!!! tip "Priors are assumptions"

    A prior states what you know before looking at the image. Choose it
    from the physics of the object or the survey, for example a magnitude
    range converted to image units, and not by looking at the fitted
    values: a prior tuned on the result makes the result circular.
