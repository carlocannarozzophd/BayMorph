# Installation

## Requirements

BayMorph requires Python 3.11 or later. Installing the package also installs
its scientific and inference dependencies: NumPy, SciPy, Astropy, Photutils
3.0 or later, Matplotlib, PyYAML, and the three supported inference backends,
Nautilus (the default), Dynesty and pocoMC, with h5py, multiprocess and
threadpoolctl. pocoMC requires PyTorch, which pip installs with it.

BayMorph typesets its figures with LaTeX: the `latex` and `dvipng` programs
of a TeX distribution must be on the `PATH`. Without them, drawing a figure
stops with an error that names the environment variable
`BAYMORPH_USETEX=0`; set it to draw the figures with Matplotlib's own
Computer Modern fonts instead:

```bash
export BAYMORPH_USETEX=0
```

`baymorph run` draws the diagnostics of the fit, unless given
`--no-diagnostics`, and a figure of every PSF it builds, as in the packaged
example; `baymorph config --advise` draws a figure of the PSF it builds.
[The BayMorph style](PLOTTING.md#the-baymorph-style) describes the
typesetting.

## Install the package

BayMorph is not on PyPI: install it from its GitHub repository. A dedicated
environment is recommended:

```bash
conda create -n baymorph -c conda-forge python=3.11 -y
conda activate baymorph

python -m pip install --upgrade pip
python -m pip install "git+https://github.com/carlocannarozzophd/BayMorph.git@main"
```

On Linux, pip may install the CUDA build of PyTorch, which takes several
gigabytes; to avoid it, install the CPU build in the new environment before
BayMorph:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
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

## Install from a clone

To install BayMorph from a clone of the repository, with the development
tools:

```bash
git clone https://github.com/carlocannarozzophd/BayMorph.git
cd BayMorph
python -m pip install -e ".[dev]"
```

## Data

BayMorph distributes no observational data: you supply your own FITS images
and choose where BayMorph writes its results. The packaged example simulates
its own image.
