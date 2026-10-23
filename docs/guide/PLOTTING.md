# Plotting

## The diagnostic figures

```python
runner.diagnostics()
```

`diagnostics()` writes three figures for each fitted model to
`fits/<model>/` in the run directory:

| File | Shows |
| --- | --- |
| `corner.pdf` | the one- and two-dimensional marginal posteriors of every parameter |
| `data_model_residual.pdf` | the data, the best model, the residual and the normalized residual |
| `radial_profile.pdf` | the elliptical radial profile of the data and of the model, with posterior bands |

The configuration chooses which figures are drawn and their format:

```yaml
diagnostics:
  format: pdf                     # or png
  corner: true
  residual: true
  radial_profile: true
  posterior_profile_bands: true   # posterior bands on the radial profile
  titles:
    single_sersic: "My galaxy"
```

`runner.corner()`, `runner.residual_maps()` and `runner.radial_profile()`
draw one figure each. All of them need a fit run in the same session; a
fit loaded from disk is plotted with the functions below. `build_psf()`
draws the diagnostics of the PSF by itself, in `psf/diagnostics/`.

## Plotting in a notebook

The functions of `baymorph.plotting` draw the same figures in a notebook
and return the Matplotlib figure, so that you can change it or save it.

**An image**, on a logarithmic scale:

```python
from baymorph.plotting import plot_log_image

fig, ax = plot_log_image(image.prepared_data, title="Prepared cutout", show=True)
```

**Data, model and residuals**, over the pixels used by the fit:

```python
import numpy as np
from baymorph.plotting import plot_log_data_model_residual

used = np.isfinite(fit.norm_residual)
data = np.where(used, fit.best_model_image + fit.residual, np.nan)

fig, axes = plot_log_data_model_residual(
    data,
    fit.best_model_image,
    norm_residual=fit.norm_residual,
    title="galaxy",
    show=True,
)
```

**The corner plot** of the posterior, with LaTeX labels for the parameters:

```python
fig, axes, _ = fit.corner()
```

or, from any posterior, including one loaded from disk:

```python
from baymorph.plotting import corner_plot

fig, axes, _ = corner_plot(
    posterior.equal_weight_samples.T,
    parameter_labels=posterior.parameter_names,
    truths=None,          # or a list of true values, for a simulation
    title="galaxy",
)
```

**The radial profile**, once `runner.radial_profile()` or
`runner.diagnostics()` has measured it:

```python
fig, axes = fit.radial_profile()
```

## The BayMorph style

Every BayMorph figure uses the same style: text typeset with LaTeX in
Computer Modern, thick axes, ticks pointing outwards with minor ticks, and
panels of 15 x 15 inches with 35-point text, which keeps the proportions of a
figure in a paper. BayMorph therefore needs a TeX distribution (`latex` and
`dvipng` on the `PATH`) to draw figures. Where LaTeX is not available, set the
environment variable `BAYMORPH_USETEX=0` (or pass `usetex=False` to
`make_plot`): the figures then use Matplotlib's own Computer Modern fonts.
Use the style for your own figures with `make_plot`, which works like
`matplotlib.pyplot.subplots`:

```python
from baymorph.plotting import make_plot

fig, ax = make_plot()                     # one panel
fig, axes = make_plot(nrows=1, ncols=3)   # three panels, axes flattened

ax.plot(radius, profile)
ax.set_xlabel(r"$R\,/\,R_e$")
```

`set_plot_style()` applies the style to Matplotlib, and `style_axes(ax)`
styles existing axes.
