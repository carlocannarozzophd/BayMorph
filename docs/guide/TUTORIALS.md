# Tutorials

Three tutorial notebooks are packaged with BayMorph. `baymorph init` copies
them into the `notebooks/` directory of a new project; they are distributed
unexecuted.

| Notebook | What it does |
| --- | --- |
| [`01_Simulated_SingleBand.ipynb`](https://github.com/carlocannarozzophd/BayMorph/blob/main/notebooks/01_Simulated_SingleBand.ipynb) | Simulates a galaxy with known parameters, fits it with the public workflow and checks that the posterior recovers the truth. The run is deliberately small: matching the truth once is a smoke test, not a convergence check. |
| [`02A_Observed_Assessment_and_Config.ipynb`](https://github.com/carlocannarozzophd/BayMorph/blob/main/notebooks/02A_Observed_Assessment_and_Config.ipynb) | Guides the observation advisor through an observed FITS image: it shows what the header holds, lets you confirm the target, prepares the image and builds its PSF, proposes every prior of the models you choose with its source, and writes one reviewed YAML configuration. It does not fit. |
| [`02B_Observed_Fit.ipynb`](https://github.com/carlocannarozzophd/BayMorph/blob/main/notebooks/02B_Observed_Fit.ipynb) | Fits the models of the configuration written by 02A, reusing the image and the PSF that 02A prepared, shows the posterior, the residuals and the evidence of each model, and writes the diagnostics. Every cell can be run again: completed products are reused, and an interrupted fit resumes from its checkpoint. |

The observed-image notebooks need a FITS image of your own: BayMorph
distributes no data. The YAML written by 02A is what 02B runs: if you change
a declaration after 02B has run, write it under a new analysis name, so that
the earlier results are kept.

A notebook for several observations is being rebuilt; until then,
[Several Observations](MULTI_OBSERVATION.md) shows the joint workflow.
