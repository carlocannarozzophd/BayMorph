# BayMorph

> \[!IMPORTANT\] **Status:** **v0.1.0** --- BayMorph has been validated
> on synthetic datasets and includes complete workflows for simulated
> galaxies, real galaxies, and multi-galaxy analyses. Scientific
> validation is ongoing.

BayMorph is an open-source Python framework for **Bayesian structural
decomposition of galaxies**.

## Official notebooks

  --------------------------------------------------------------------------------------------------------------------------
  Notebook                                                                               Description
  -------------------------------------------------------------------------------------- -----------------------------------
  [`01_Simulated_Galaxy_Analysis.ipynb`](notebooks/01_Simulated_Galaxy_Analysis.ipynb)   Complete workflow on a simulated
                                                                                         galaxy.

  [`02_Real_Galaxy_Analysis.ipynb`](notebooks/02_Real_Galaxy_Analysis.ipynb)             Workflow on a single real galaxy.

  [`03_Multi_Galaxy_Analysis.ipynb`](notebooks/03_Multi_Galaxy_Analysis.ipynb)           Automatically processes all FITS
                                                                                         images in a directory.
  --------------------------------------------------------------------------------------------------------------------------

## Configuration templates

``` text
configs/
├── simulated_galaxy.yaml
└── real_galaxy.yaml
```

## Validation

-   synthetic galaxy recovery
-   complete simulated-galaxy workflow
-   complete real-galaxy workflow
-   automated unit and integration tests

``` bash
pytest
```

**159 tests passed**
