# BayMorph Documentation

This directory contains the technical documentation for BayMorph.

BayMorph is a Bayesian framework for the morphological modelling of astronomical images. The repository `README.md` provides the user-facing overview, installation instructions, quick-start examples, and links to the official notebooks. The documents in this directory describe the architecture, validation strategy, and long-term design decisions in greater detail.

---

# Documents

## `ARCHITECTURE.md`

Describes the internal architecture of BayMorph, including

- design philosophy;
- public and internal APIs;
- package organization;
- configuration and output architecture;
- pipeline workflow;
- model architecture;
- PSF construction;
- Bayesian inference;
- plotting system.

Read this document if you want to understand how BayMorph is organized internally or where new functionality should be implemented.

---

## `VALIDATION.md`

Describes the scientific validation strategy, including

- synthetic-galaxy validation;
- end-to-end pipeline validation;
- real-galaxy demonstration on TARGET;
- purpose of the official notebooks;
- current validation status;
- known limitations and future validation plans.

Read this document if you want to understand what has been validated and what remains under active development.

---

# Related top-level documents

## `README.md`

Repository overview, installation instructions, quick-start guide, and links to the official notebooks.

---

## `ROADMAP.md`

Near-term, medium-term, and long-term development plans.

---

## `DECISIONS.md`

Collection of accepted architectural decisions, including

- Bayesian model comparison;
- Dynesty as the inference engine;
- source-masked noise estimation;
- empirical PSF construction;
- pipeline organization;
- output structure;
- public API design.

---

# Recommended reading order

## For users

1. `README.md`
2. `notebooks/01_Simulated_Galaxy_Analysis.ipynb`
3. `notebooks/02_Real_Galaxy_Analysis.ipynb`

---

## For developers

1. `docs/ARCHITECTURE.md`
2. `docs/VALIDATION.md`
3. `DECISIONS.md`
4. `ROADMAP.md`