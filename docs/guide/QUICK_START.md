# Quick Start

BayMorph analyses are defined by YAML configuration files and run either
from the command line or from Python. Every option of the command line is
listed in [Command-Line Interface](CLI.md), and every configuration key in
[Configuration Reference](CONFIGURATION.md).

!!! warning "Status of the real-data path"

    The observation model of the real-data path is under revision. The
    effective-PSF rendering of AD-046 is implemented; the noise model of
    AD-047 is accepted and being implemented. No real-data fit has yet been
    validated scientifically. The current priority is stated in the
    [roadmap](https://github.com/carlocannarozzophd/BayMorph/blob/main/ROADMAP.md).

## Create a workspace

```bash
baymorph init my_analysis
cd my_analysis
```

The workspace contains `configs/` with the example configuration,
`notebooks/` with the tutorial notebooks, and empty `data/` and `runs/`
directories. It is a convenience: BayMorph also runs in any directory.

## Run the simulated example

```bash
baymorph run configs/simulated_galaxy.yaml
```

The example simulates a 121 x 121 px image of one Sérsic galaxy with a
Gaussian PSF and Gaussian noise, prepares it, fits a single Sérsic model with
Nautilus and writes the diagnostics. Its products are written under
`runs/simulated_galaxy/`; [Preparation and Products](PREPARATION_AND_PRODUCTS.md)
describes them.

## Fit an observed image

Generate a configuration from a FITS image. A target seed and a flux prior
are required, because BayMorph does not derive either from the image:

```bash
baymorph config data/galaxy.fits \
  --target-x 500 \
  --target-y 500 \
  --model single_sersic \
  --uniform-prior flux 100 1000000 \
  --output configs/galaxy.yaml
```

Review the file before running it. Its priors are in the units of the image,
and two defaults are not scaled to them: the prior of the residual
background, `fit.background.priors.background_level`, uniform between −50
and 50, and the brightest peak accepted for a PSF star,
`psf.empirical.candidate_selection.max_peak`, 80000. Then run it:

```bash
baymorph run configs/galaxy.yaml
```

The run prepares the image, selects the PSF stars, builds the empirical PSF,
fits the model and writes the diagnostics. `--no-diagnostics` skips the
diagnostics.

## The same workflow from Python

```python
from baymorph import BayMorphRunner

runner = BayMorphRunner("configs/simulated_galaxy.yaml")
result = runner.run(diagnostics=True)
```

`run()` executes the stages in order. They can also be run one at a time:

```python
runner = BayMorphRunner("configs/galaxy.yaml")

runner.prepare_image()
runner.select_psf_candidates()
runner.build_psf()
result = runner.fit()
runner.diagnostics()
```

`simulate()` comes first, and only for a configuration with a `simulation`
section. `select_psf_candidates()` and `build_psf()` apply only to an
empirical PSF and raise an error for an analytic one, such as `gaussian`.

## Repeating and resuming a workflow

Image preparation, PSF candidate selection and PSF construction reuse a
completed product of the same output namespace instead of rebuilding it
(AD-042), after checking that its inputs still match the configuration. An
interrupted workflow therefore continues by running it again.

A completed fit is not reused: when its completion marker exists and
`outputs.overwrite` is false, `fit()` raises `FileExistsError`. A fit
interrupted before completion resumes only from a signed checkpoint, which
Nautilus (`fit.sampler.init.filepath`) and pocoMC
(`fit.sampler.run.resume_state_path`) support; both accept `auto`. Dynesty
has no checkpoint. A checkpoint whose signature does not match the current
problem is refused.
