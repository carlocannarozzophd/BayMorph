# Several Observations

When the same galaxy is observed several times, in different bands, with
different instruments or at different epochs, BayMorph can fit all the
images together. Each image keeps its own pixel grid, PSF, mask, noise and
residual background; the model parameters can be shared between images or
left free in each. A single image is simply the one-image case of the same
analysis.

## How parameters are tied

For every parameter you choose a *coupling*:

| Coupling | Meaning | Typical use |
| --- | --- | --- |
| `shared` | one value for all images | the centre; the shape, when it is the same in every band |
| `independent` | one value per image, each with its own prior | the flux; the residual background, which is always independent |
| `hierarchical` | one value per image, correlated with the others through a correlation inferred from the data | a radius or an index that changes little between bands |
| `wavelength` | like `hierarchical`, with a correlation that decreases with the distance in wavelength | a property that varies smoothly with wavelength |

With `hierarchical` and `wavelength`, the images share information without
being forced to agree: if they agree, the values are pulled together; if
they do not, they stay apart. Each image's value keeps exactly the prior you
declare. [Hierarchical Models](../theory/HIERARCHICAL_MODELS.md) explains
how.

## Sky coordinates

With `parameter_frame: sky`, which `make_multi_parameters` and
`make_multi_observation_config` write by default, the joint parameters live
on the sky, so that images with different pixel grids describe the same
galaxy (a joint configuration that does not declare `parameter_frame` uses
`detector`, the pixel parameters of the model):

| Parameter | Meaning | Units |
| --- | --- | --- |
| `ra`, `dec` | the centre | degrees |
| `re_arcsec` | the effective radius | arcseconds |
| `position_angle` | the major axis, measured from local east towards local north | radians |

The other parameters, such as `n` and `q`, keep their meaning. Every image
needs a celestial WCS in its FITS header. Note that `position_angle` is not
the astronomical position angle, which is measured from north through east.

## Building the configuration

A joint analysis has one configuration per image and one joint
configuration that ties them. `make_multi_parameters` builds the coupling of
every parameter, and `make_multi_observation_config` writes all the files:

```python
from baymorph import make_multi_observation_config, make_multi_parameters

observations = {
    "g": "data/galaxy_g.fits",
    "r": "data/galaxy_r.fits",
}

parameters = make_multi_parameters(
    "single_sersic",
    observations,
    target_centers={"g": (100, 100), "r": (100, 100)},
    prior_bounds={
        "flux": (1.0e3, 1.0e6),
        "re_arcsec": (0.5, 5.0),
        "background_level": (-10.0, 10.0),
    },
    shared=["ra", "dec", "re_arcsec", "n", "q", "position_angle"],
)

make_multi_observation_config(
    observations,
    target_positions={"g": {"x": 100, "y": 100}, "r": {"x": 100, "y": 100}},
    parameters=parameters,
    output_dir="configs/joint",
    analysis_name="galaxy_joint",
)
```

Parameters not listed in `shared` are `independent`. `prior_bounds` gives
uniform priors; parameters without one keep their defaults, and a parameter
with no default, such as the flux, must have one, as must a shared
`re_arcsec`. The function writes `configs/joint/g.yaml`,
`configs/joint/r.yaml` and `configs/joint/joint.yaml`.
`observation_options` sets the PSF, the cutout or the likelihood of each
image, for example
`{"g": {"psf_type": "gaussian", "gaussian_fwhm_pixels": 3.0}}`.

The joint file lists the images and the coupling of every parameter:

```yaml
observations:
  g:
    config_path: g.yaml
  r:
    config_path: r.yaml

outputs:
  root: runs
  analysis_name: galaxy_joint

fit:
  models: [single_sersic]
  background:
    model: constant
  multi_observation:
    parameter_frame: sky
    parameters:
      flux:
        coupling: independent
        priors:
          g: {type: uniform, lower: 1000, upper: 1000000}
          r: {type: uniform, lower: 1000, upper: 1000000}
      n:
        coupling: shared
        prior: {type: uniform, lower: 0.5, upper: 6}
      # ...one entry for every parameter
```

To make a parameter hierarchical, write `coupling: hierarchical` with one
`prior`, the prior of each image's value. For `coupling: wavelength`, also
give each image its wavelength:

```yaml
observations:
  g:
    config_path: g.yaml
    covariates:
      wavelength: {value: 477, unit: nm}
  r:
    config_path: r.yaml
    covariates:
      wavelength: {value: 623, unit: nm}
```

BayMorph never guesses a wavelength, a filter or a coupling from the name of
an image: `g` and `r` are just labels.

## Running it

The same runner runs a joint analysis:

```python
from baymorph import BayMorphRunner

runner = BayMorphRunner("configs/joint/joint.yaml")
result = runner.run()
```

or `baymorph run --no-diagnostics configs/joint/joint.yaml`. The runner
prepares every image and builds every PSF in the run directory of that
image, then fits them together. The joint posterior and the evidence are
written once, under `runs/galaxy_joint/fits/<model>/`; the model and the
residuals of each image are written under `observations/<name>/` in the
same directory.

!!! warning "Diagnostics of a joint analysis"

    `runner.diagnostics()`, and so `runner.run(diagnostics=True)` and
    `baymorph run` without `--no-diagnostics`, look for the configuration
    of each image in the working directory instead of beside the joint
    file, and stop when it is not there. Draw the figures from the path of
    the joint file instead:

    ```python
    from baymorph.pipeline import (
        run_corner_diagnostics,
        run_radial_profile_diagnostics,
        run_residual_map_diagnostics,
    )

    for draw in (run_corner_diagnostics, run_residual_map_diagnostics,
                 run_radial_profile_diagnostics):
        draw("configs/joint/joint.yaml", result)
    ```

    The corner plot is written under `fits/<model>/`, and the figures of
    each image under `observations/<name>/`.

## Results

A shared parameter has one name; an independent one has one name per image,
with the name of the image after two underscores:

```python
result.posterior.parameter_names
# ['flux__g', 'flux__r', 're_arcsec', 'n', 'q', 'position_angle',
#  'ra', 'dec', 'background_level__g', 'background_level__r']

result.posterior.summary("re_arcsec")      # one radius for both bands
result.logz, result.reduced_chi2

result.observations["g"].best_model_image  # the model of the g image
result.observations["g"].norm_residual     # its normalized residual
```

`result.posterior` is the same `InferenceResult` as for one image (see
[Results](RESULTS.md)), and `result.observations` holds, for each image,
the best model, the residuals and the parameters in the pixel frame of that
image. A hierarchical parameter appears in the posterior through its sampled
coordinates, `<name>__rho` (`<name>__rstar` with `coupling: wavelength`) and
one latent value `<name>__z__<image>` per image; its value in each image at
the best sample is in `result.observations`.

## Limits

- The noise of different images is assumed independent: the joint
  likelihood is the product of the likelihoods of the images.
- Each image declares its own effective gain, in its own configuration.
- The six models with bars, rings, spiral arms, broken discs, boxy
  isophotes or cores are single-image models: a joint analysis refuses them.
- A joint analysis fits one model. The configuration of an image may not
  declare `data.hdu` or `data.calibration`: each image is the one BayMorph
  selects by default in its file, and its header is read as it is.
- The posterior holds the latent coordinates of a hierarchical or
  wavelength parameter: BayMorph does not yet turn its samples into the
  value of that parameter in each image.
