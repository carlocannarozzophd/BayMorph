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
takes about a minute and a half on a laptop. Its figures are typeset with
LaTeX: without `latex` and `dvipng`, set `BAYMORPH_USETEX=0` first, as
[Installation](INSTALLATION.md#requirements) explains. The runner prints one
line per step:

```text
✓ Synthetic observation generated
✓ Image preparation completed
✓ Ready: correlated-noise calibration
✓ Ready: nearest-neighbour factorization
...
Completed Nautilus fit: single_sersic
...
✓ Bayesian fitting completed
✓ Corner plots generated
✓ Residual maps generated
✓ Radial profiles generated
✓ Configured diagnostics generated
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
...

Best-fit parameters
flux = 20004.168
re = 6.0102793
n = 2.0188553
q = 0.75213613
...
```

The simulated galaxy has `flux` 20000, `re` 6 px, `n` 2 and `q` 0.75, and a
reduced chi-square close to 1 says that the model describes the image down
to the noise.

## 4. Fit your own image

Copy a FITS image into `data/` and give BayMorph the centre of the galaxy,
in pixels or on the sky, as a catalogue gives it. The observation advisor
reads the header, prepares the
image, builds the PSF from its stars and proposes a configuration, with the
source of every prior:

```bash
baymorph config data/galaxy.fits \
  --target-ra 150.1 --target-dec 2.2 \
  --model single_sersic \
  --advise \
  --output configs/galaxy.yaml
```

Read what it printed. The header report says what BayMorph has read from
the header and what it could not establish; declare what is missing with
the options of [Opening an Image](OPENING.md#declaring-what-the-header-lacks),
such as `--zeropoint`. The proposal lists every prior with its source;
[Writing a Configuration](CONFIGURING.md) says which choices are yours. When
the proposal is right, run the same command with `--accept`, which writes
the configuration, then fit:

```bash
baymorph config data/galaxy.fits \
  --target-ra 150.1 --target-dec 2.2 \
  --model single_sersic \
  --advise --accept \
  --output configs/galaxy.yaml

baymorph run configs/galaxy.yaml
```

The fit reuses the image, the masks and the PSF that the advisor prepared,
fits the model and draws the diagnostics. Without `--advise`,
`baymorph config` writes the configuration at once, and every flux prior
must then be declared with `--uniform-prior`, in the units of the image.

Without `--offline`, the advisor may ask the ACS Zeropoint Calculator of
STScI for the zeropoint of a drizzled ACS image whose header gives no usable
one; with it, the advisor asks no service. When neither the header nor the
sky gives the effective gain of the noise model, the proposal says
`Effective gain: not proposed`: `--accept` still writes the configuration,
but the fit stops before sampling until you declare
`fit.likelihood.effective_gain` in it, as
[Sky and Noise](SKY_AND_NOISE.md#the-effective-gain) explains.

If the preparation stops because the cutout does not contain the plateau of
the galaxy and its sky margin, declare a fixed cutout large enough to hold
them, with `--cutout-mode fixed` and `--cutout-size` followed by its side in
pixels; see [Target and Cutout](IMAGE.md#the-cutout).

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
