# Output Files

Every product of an analysis is written under `outputs.root/object.name/`,
the run directory. Each product is written once and then reused: the
preparation and the PSF by later runs of the same configuration, and a
finished fit is never overwritten unless `outputs.overwrite` is true.

```text
runs/galaxy/
├── config_resolved.yaml        the complete configuration, defaults included
├── runtime_report.yaml         the configuration and what preparation measured
├── image_preparation/          the cutout, masks, sky and noise
├── psf/                        the empirical PSF
├── fits/
│   └── single_sersic/          one directory per fitted model
└── model_comparison/           the Bayes factor, when enabled
```

## `image_preparation/`

| File | Content |
| --- | --- |
| `prepared_image.fits` | the cutout with the sky subtracted: the data of the fit |
| `noise_map.fits` | the noise scale of the sky in every pixel |
| `fitting_mask.fits` | the mask of the fit, non-zero for excluded pixels |
| `mask.fits` | the automatic mask of the detected sources |
| `segmentation.npy`, `target_segment.npy` | the detected sources and the target |
| `radial_profile.npz`, `sky_region.npz` | the radial profile of the target and the sky region |
| `image_preparation_metadata.json` | the cutout, the target, the subtracted sky level, the plateau, the sky gradient, the contaminant rule with the segments it left unmasked, and the growth of the masks |
| `correlated_gaussian_noise.npz` | the correlation of the sky, measured before the first fit, and the estimator that measured it |
| `correlated_gaussian_vecchia/m24/` | the nearest-neighbour factorization of that correlation used by the likelihood |
| `IMAGE_PREPARATION_COMPLETE.json` | written last, when every product is valid |

The directory also holds intermediate arrays used by the diagnostics.

## `psf/`

| File | Content |
| --- | --- |
| `psf.fits` | the PSF kernel; its `SUBPIXEL` extension holds the kernel at eleven sub-pixel phases per axis |
| `psf_metadata.json` | how the kernel was built, and its properties |
| `psf_candidates.json` | the stars selected for the PSF |
| `diagnostics/` | the stamps of the stars and the profile of the kernel |

## `fits/<model>/`

| File | Content |
| --- | --- |
| `posterior.npz` | the weighted posterior samples: the result of the fit; load it with `InferenceResult.load` |
| `best_parameters.json` | the parameters of the sample of highest likelihood |
| `fit_summary.json`, `fit_summary.txt` | the evidence, the goodness of fit, the sampler and the noise model used |
| `best_model_image.npy` | the model image at the best sample |
| `residual.npy` | data minus model |
| `norm_residual.npy` | the residual divided by the noise of each pixel, `NaN` on masked pixels |
| `noise.npy` | the noise scale of the sky |
| `config_resolved.yaml` | the configuration of this fit |
| `corner.pdf`, `data_model_residual.pdf`, `radial_profile.pdf` | the diagnostic figures, in the format of `diagnostics.format` |
| `nautilus_checkpoint.hdf5`, `pocomc_state.h5` | the checkpoint, when `auto` was requested, with its signature in a `.target.json` file |
| `COMPLETE.json` | written last, when every product is valid |

With `fit.comparison.enabled`, `model_comparison/bayes_factor.json` holds the
Bayes factor of `double_sersic` over `single_sersic`.

## Simulations

A configuration with a `simulation` section also writes the synthetic image
to `data.image_path`, usually under `input/` in the run directory, and the
true parameters to `truth/truth.json`. The corner plot marks the truth when
that file exists.

## Joint analyses

A joint analysis writes the preparation and the PSF of every image in the
run directory of that image, and its own products under
`outputs.root/outputs.analysis_name/`:

```text
runs/galaxy_joint/
└── fits/
    └── single_sersic/
        ├── posterior.npz, best_parameters.json, fit_summary.json
        ├── corner.pdf
        └── observations/
            ├── g/    best_model_image.npy, residual.npy, norm_residual.npy,
            │         fit_summary.json, data_model_residual.pdf, radial_profile.pdf
            └── r/    ...
```

The posterior, the evidence and the corner plot belong to the joint fit and
exist once; the images and residuals belong to each observation.
