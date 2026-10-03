# Simulated Images

A simulated galaxy, whose parameters you know, is the way to test a model, a
prior or a configuration before trusting it on real data.

## A synthetic image in Python

```python
import numpy as np
from baymorph import SyntheticSersic, make_observation

galaxy = SyntheticSersic(
    flux=2.0e4, re=6.0, n=2.0, q=0.75,
    theta=np.deg2rad(25.0),          # radians
    x0=100.0, y0=100.0,              # pixels
)

observation = make_observation(
    galaxy,
    shape=(201, 201),                # (ny, nx)
    sky=20.0,                        # constant sky level
    psf_fwhm=2.0,                    # Gaussian PSF, pixels
    poisson_noise=False,
    read_noise_sigma=2.0,
    seed=42,
)

observation.to_fits("data/simulated.fits")
```

`make_observation` renders the galaxy with the same Sérsic model BayMorph
fits, convolves it with a Gaussian PSF, adds the sky and the noise, and
returns every stage: `observation.intrinsic_image`,
`observation.convolved_image`, `observation.image`, the final noisy image,
and `observation.truth`, the true parameters. `flux` is the total flux of
the profile, of which the image holds the part that falls inside it.

## A simulation in the configuration

A configuration with a `simulation` section makes its own image before
fitting it. The packaged example, `configs/simulated_galaxy.yaml`, written
by `baymorph init`, is one:

```yaml
simulation:
  shape: [201, 201]
  seed: 42
  truth: {flux: 20000.0, re: 6.0, n: 2.0, q: 0.75, theta_deg: 25.0,
          x0: 100.0, y0: 100.0}
  psf: {type: gaussian, fwhm: 2.0}
  noise: {sky: 20.0, poisson_noise: false, read_noise: 2.0}
```

Here the angle is given in degrees, `theta_deg`. `runner.simulate()`, or
`baymorph run`, writes the image to `data.image_path` and the truth to
`truth/truth.json` in the run directory, and the corner plot of the fit
marks the true values.

```python
runner = BayMorphRunner("configs/simulated_galaxy.yaml")
simulation = runner.simulate()
runner.prepare_image()
fit = runner.fit()
```

## Did the fit recover the truth?

Compare each true value with its credible interval. The true `x0` and `y0`
are in pixels of the full image, the fitted ones in pixels of the cutout:
shift them by the origin of the cutout first.

```python
truth = runner.simulation.truth

for name in ("flux", "re", "n", "q"):
    print(name, truth[name], fit.posterior.summary(name))
```

One run is a check, not a test: with 68 percent intervals, about one
parameter in three falls outside its interval by chance. Because the flux,
the radius and the index are correlated, judge them together with
`fit.posterior.mahalanobis_distance(truth)` (see [Results](RESULTS.md)),
and repeat the simulation with different noise seeds before concluding that
a model is biased.

The sky of the simulation is removed by the preparation like the sky of a
real image, so the fitted `background_level` is the residual left after
that, close to zero, and not the simulated sky level.
