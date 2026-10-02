# Tutorials

Three tutorial notebooks are packaged with BayMorph. `baymorph init` copies
them into the `notebooks/` directory of a new project; they are distributed
unexecuted.

| Notebook | What it does |
| --- | --- |
| [`01_Simulated_SingleBand.ipynb`](https://github.com/carlocannarozzophd/BayMorph/blob/main/notebooks/01_Simulated_SingleBand.ipynb) | Simulates a galaxy with known parameters, fits it with the public workflow and checks that the posterior recovers the truth. The run is deliberately small: matching the truth once is a smoke test, not a convergence check. |
| [`02A_Observed_Assessment_and_Config.ipynb`](https://github.com/carlocannarozzophd/BayMorph/blob/main/notebooks/02A_Observed_Assessment_and_Config.ipynb) | Inspects an observed FITS image with the observation advisor, lets you confirm the target and declare the scientific assumptions, and writes one reviewed YAML configuration. It neither prepares the image nor fits it. |
| [`02B_Observed_Fit.ipynb`](https://github.com/carlocannarozzophd/BayMorph/blob/main/notebooks/02B_Observed_Fit.ipynb) | Runs the whole pipeline on the configuration written by 02A: preparation, empirical PSF, fit and diagnostics. Every cell can be run again: completed products are reused, and an interrupted fit resumes from its checkpoint. |

The observed-image notebooks need a FITS image of your own: BayMorph
distributes no data. The YAML written by 02A is the contract between the two
notebooks; a changed declaration is a different scientific problem and
belongs in a new output namespace.

The multi-observation tutorial is being rebuilt against the current public
API and is not yet distributed. Until then, the joint workflow is described
in [Multi-Observation Workflow](../architecture/MULTI_OBSERVATION_PUBLIC_WORKFLOW.md).

The notebooks under `validation/` in the repository are research
demonstrations of the validation campaigns, not tutorials.
