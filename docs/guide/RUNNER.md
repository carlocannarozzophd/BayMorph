# The Runner

`BayMorphRunner` runs a BayMorph analysis from its configuration. It is the
only object most analyses need.

## Run everything

```python
from baymorph import BayMorphRunner

runner = BayMorphRunner("configs/galaxy.yaml")
fit = runner.run(diagnostics=True)
```

`run()` executes the stages in order and returns the fit products, or, when
`fit.models` lists several models, a dictionary of them by model name;
`diagnostics=True` also writes the diagnostic figures. The
command line does the same, figures included unless `--no-diagnostics` is
given:

```bash
baymorph run configs/galaxy.yaml
```

## Run one stage at a time

In a notebook it is often clearer to run the stages separately and look at
each product before going on:

```python
runner = BayMorphRunner("configs/galaxy.yaml")

image = runner.prepare_image()        # cutout, masks, sky and noise
runner.select_psf_candidates()        # stars for the empirical PSF
psf = runner.build_psf()              # the PSF kernel
fit = runner.fit()                    # the posterior
runner.diagnostics()                  # corner plot, residuals, profile
```

| Method | Returns | Guide page |
| --- | --- | --- |
| `simulate()` | the synthetic observation; only for a configuration with a `simulation` section, and before `prepare_image()` | [Simulated Images](SIMULATIONS.md) |
| `prepare_image()` | the prepared observation: cutout, masks, sky, noise | [Target and Cutout](IMAGE.md) |
| `select_psf_candidates()` | the list of selected stars | [Point-Spread Function](PSF.md) |
| `build_psf()` | the PSF kernel, a 2D array | [Point-Spread Function](PSF.md) |
| `fit()` | the fit products; a dictionary by model name for several models | [Fitting](FITTING.md), [Results](RESULTS.md) |
| `diagnostics()` | writes the figures enabled in the configuration | [Plotting](PLOTTING.md) |
| `corner()`, `residual_maps()`, `radial_profile()` | write one diagnostic figure each | [Plotting](PLOTTING.md) |

`select_psf_candidates()` and `build_psf()` apply only to an empirical PSF.
With an analytic PSF, such as `psf.type: gaussian`, skip them.

After a stage has run, its product is also available as an attribute:
`runner.image`, `runner.psf_candidates`, `runner.psf`, `runner.results` and,
for a simulation, `runner.simulation`. `runner.run_dir` is the directory of
the analysis, `outputs.root/object.name/`.

## Where files go

Every product of an analysis is written under `outputs.root/object.name/`.
When the configuration file lives in a directory called `configs/`, as in a
project created by `baymorph init`, the relative paths `data.image_path` and
`outputs.root` are read relative to the project directory, the parent of
`configs/`, so the same configuration works from a notebook in
`notebooks/` and from a terminal in the project. The files themselves are
listed in [Output Files](../reference/OUTPUT_FILES.md).

## Running again

The preparation and PSF stages can be run again at no cost: when the
products of the same configuration already exist and still match their
inputs, the runner loads them and says so instead of rebuilding them. An
interrupted analysis therefore continues by running it again.

A finished fit is never overwritten by accident: with
`outputs.overwrite: false`, the default, `fit()` raises `FileExistsError`
when the model has already been fitted. To fit again, set
`outputs.overwrite: true`, or, better, give the analysis a new
`object.name`, so that the earlier results stay as they are.

A fit interrupted before it finished resumes from its checkpoint when the
sampler keeps one: set `fit.sampler.init.filepath: auto` for Nautilus, or
`fit.sampler.run.resume_state_path: auto` for pocoMC. A checkpoint made with
different data, priors or sampler settings is refused rather than resumed.
Dynesty keeps no checkpoint.

!!! note "Changing the configuration"

    The image, the cutout, the masks, the PSF and the priors define the
    scientific problem. When you change one of them, write the analysis
    under a new `object.name`: products made with the old settings are then
    kept, and never mixed with the new ones.

## Quiet mode

The runner prints one line per completed stage. To silence it:

```python
runner = BayMorphRunner("configs/galaxy.yaml", verbose=False)
```

A configuration can also be passed as a Python dictionary instead of a path.
