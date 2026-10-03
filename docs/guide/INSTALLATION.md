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
| `interactive` | anywidget, IPython, ipywidgets, Plotly | interactive tools in notebooks, such as clicking on the target |
| `dev` | pytest, Jupyter, ipykernel, Black, Ruff, scikit-learn, build, Twine | development and tests |

```bash
python -m pip install "baymorph[interactive] @ git+https://github.com/carlocannarozzophd/BayMorph.git@main"
```

The [tutorial notebooks](TUTORIALS.md) also need Jupyter, for example
`python -m pip install jupyterlab`.

## Data

BayMorph distributes no observational data: you supply your own FITS images
and choose where BayMorph writes its results. The packaged example simulates
its own image.

To install BayMorph from a clone of the repository, for development, see
[For Developers](../DEVELOPERS.md).
