# Quick Start

This page runs a first fit in a few minutes: first on a simulated galaxy
packaged with BayMorph, then on an image of your own.

## 1. Create a project

```bash
baymorph init my_analysis
cd my_analysis
```

The project has four directories: `configs/`, with an example
configuration, `notebooks/`, with the tutorial notebooks, and empty
`data/` and `runs/`. BayMorph also runs in any other directory; the
project is only a convenient layout.

## 2. Fit a simulated galaxy

```bash
baymorph run configs/simulated_galaxy.yaml
```

The example makes a 201 x 201 px image of one Sérsic galaxy, with a
Gaussian PSF and Gaussian noise, and fits it with a single Sérsic model. It
takes about a minute and a half on a laptop. The runner prints one line per
step:

```text
✓ Synthetic observation generated
✓ Image preparation completed
✓ Ready: correlated-noise calibration
✓ Ready: nearest-neighbour factorization
...
Completed Nautilus fit: single_sersic
✓ Corner plots generated
✓ Residual maps generated
✓ Radial profiles generated
BayMorph analysis completed.
```

## 3. Look at the results

Everything is in `runs/simulated_galaxy/`. The fit of the model is in
`fits/single_sersic/`:

| File | Content |
| --- | --- |
| `fit_summary.txt` | the evidence, the goodness of fit and the best parameters |
| `corner.pdf` | the posterior of every parameter, with the true values marked |
| `data_model_residual.pdf` | the image, the model and the residuals |
| `radial_profile.pdf` | the radial profile of the image and of the model |
| `posterior.npz` | the weighted posterior samples |

`fit_summary.txt` reports, among other things, the statistics and the best
parameters; your numbers may differ in the last digits:

```text
Fit statistics
chi2 = 40071.934
ndof = 40368
reduced_chi2 = 0.99266583
logZ = -85483.241

Best-fit parameters
flux = 20004.168
re = 6.0102793
n = 2.0188553
q = 0.75213613
```

The simulated galaxy has `flux` 20000, `re` 6 px, `n` 2 and `q` 0.75, and a
reduced chi-square close to 1 says that the model describes the image down
to the noise.

## 4. Fit your own image

Copy a FITS image into `data/` and write a configuration for it. Tell
BayMorph where the galaxy is and give a prior range for its flux, in the
units of the image:

```bash
baymorph config data/galaxy.fits \
  --target-x 500 \
  --target-y 500 \
  --model single_sersic \
  --uniform-prior flux 100 1000000 \
  --output configs/galaxy.yaml
```

Open `configs/galaxy.yaml` and review it before running: the priors, the
residual background and the brightest star accepted for the PSF are in the
units of your image, and [Writing a Configuration](CONFIGURING.md) lists
what to check. Then:

```bash
baymorph run configs/galaxy.yaml
```

BayMorph cuts out the galaxy, masks the other sources, subtracts the sky,
builds the PSF from the stars of the image, fits the model and draws the
diagnostics.

## 5. The same from Python

```python
from baymorph import BayMorphRunner

runner = BayMorphRunner("configs/galaxy.yaml")
fit = runner.run(diagnostics=True)

for name in fit.posterior.parameter_names:
    print(name, fit.posterior.summary(name))
```

## Next steps

- The [User Guide](index.md) shows each step on its own: the cutout, the
  masks, the sky and the noise, the PSF, the fit and the results.
- The [Models](../models/index.md) section describes every model you can
  fit.
- The [tutorial notebooks](TUTORIALS.md) walk through a simulated and an
  observed galaxy.
