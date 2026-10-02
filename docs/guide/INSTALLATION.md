# Installation

## Requirements

BayMorph requires Python 3.11 or later. Installing the package also installs
its scientific and inference dependencies: NumPy, SciPy, Astropy, Photutils
3.0 or later, Matplotlib, PyYAML, and the three supported inference backends,
Nautilus (the default), Dynesty and pocoMC, with h5py, multiprocess and
threadpoolctl.

## Install the package

A dedicated environment is recommended:

```bash
conda create -n baymorph -c conda-forge python=3.11 -y
conda activate baymorph

python -m pip install --upgrade pip
python -m pip install "git+https://github.com/carlocannarozzophd/BayMorph.git@main"
```

Verify the installation:

```bash
baymorph --version
baymorph --help
```

## Optional extras

| Extra | Installs | Use |
| --- | --- | --- |
| `interactive` | anywidget, IPython, ipywidgets, Plotly | interactive inspection tools in notebooks |
| `dev` | pytest, Jupyter, ipykernel, Black, Ruff, scikit-learn, build, Twine | development and tests |

## Development installation

```bash
git clone https://github.com/carlocannarozzophd/BayMorph.git
cd BayMorph

python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

The quality gates for contributors are described in
[`CONTRIBUTING.md`](https://github.com/carlocannarozzophd/BayMorph/blob/main/CONTRIBUTING.md).

## Build this documentation locally

```bash
python -m pip install -r requirements-docs.txt
python -m mkdocs serve
```

`python -m mkdocs build --strict` builds the site and fails on any warning.

## What the distribution contains

The package contains software, generic example configurations, tutorial
notebooks and tests. It contains no observational data and no generated
results such as run directories, posteriors, benchmark tables or figures.
Users supply their own FITS images and choose where BayMorph writes its
products.
