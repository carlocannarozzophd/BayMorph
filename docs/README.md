# BayMorph Documentation

This directory contains the technical documentation for BayMorph.

BayMorph is a Bayesian framework for morphological modelling of astronomical images. The main repository README provides the user-facing overview, installation instructions, quick-start example, and links to the tutorial notebooks. The documents in this directory describe the architecture and validation strategy in more detail.

---

## Documents

### `ARCHITECTURE.md`

Describes the internal structure of BayMorph, including:

- design philosophy
- public and internal APIs
- package organization
- pipeline architecture
- model, PSF, fitting, and plotting components
- run-output organization

Read this document if you want to understand how the package is structured or where new functionality should be added.

---

### `VALIDATION.md`

Describes the validation strategy, including:

- synthetic single-Sérsic validation
- end-to-end mock-pipeline validation
- real-galaxy demonstration on TARGET
- the purpose of each official notebook
- current validation limits and next steps

Read this document if you want to understand what has been tested and what remains under active validation.

---

## Related top-level documents

### `ROADMAP.md`

Summarizes near-term, medium-term, and long-term development plans.

### `DECISIONS.md`

Records accepted architectural decisions, including the choice of Bayesian evidence, `dynesty`, source-masked noise estimation, diagnostic products, `configs/`, and `runs/`.

---

## Recommended reading order

For new users:

1. `README.md`
2. `notebooks/01_SingleSersic_Validation.ipynb`
3. `notebooks/02_MockPipeline_Validation.ipynb`
4. `notebooks/TARGET_BayMorph_Test.ipynb`

For developers:

1. `docs/ARCHITECTURE.md`
2. `docs/VALIDATION.md`
3. `DECISIONS.md`
4. `ROADMAP.md`
