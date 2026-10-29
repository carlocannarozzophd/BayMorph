# Results

`fit()` returns the fit products of the model. The posterior is in
`fit.posterior`:

```python
fit = runner.fit()
posterior = fit.posterior

posterior.parameter_names
# ['flux', 're', 'n', 'q', 'theta', 'x0', 'y0', 'background_level']
```

## Parameter estimates

```python
posterior.summary("re")
# {'q16': 14.80, 'median': 15.01, 'q84': 15.22}
```

`summary()` gives the weighted median and the 16th and 84th percentiles of
the marginal posterior, the usual 68 percent credible interval. On the
example image, whose galaxy has a true effective radius of 15 px, the fit
gives `re` = 15.01 (+0.21/−0.21) px.

A table of every parameter:

```python
from astropy.table import Table

Table(rows=[
    {"parameter": name, **posterior.summary(name)}
    for name in posterior.parameter_names
])
```

| Attribute | Content |
| --- | --- |
| `posterior.median_parameters` | the posterior median of every parameter |
| `posterior.best_parameters` | the sample of highest likelihood |
| `posterior.samples`, `posterior.weights` | the weighted posterior samples and their normalized weights |
| `posterior.equal_weight_samples` | samples of equal weight, derived from the weighted ones, for plotting |
| `posterior.effective_sample_size` | the effective number of independent samples |

Units follow the parameter: fluxes and the background in image units, radii
and positions in pixels of the cutout, angles in radians. The
[Models](../models/index.md) pages give each one.

!!! note "Weighted samples"

    A nested sampler returns samples with weights. Summaries, intervals
    and evidences are computed from the weighted samples, which are the
    result of the fit; the equal-weight samples are a convenience for
    plots.

## Evidence and goodness of fit

```python
posterior.logz, posterior.logz_error   # natural log of the evidence
fit.reduced_chi2, fit.ndof             # 0.994 with 62160 degrees of freedom
```

The log-evidence is used to compare models fitted to the same image (see
[Bayesian Inference](../theory/BAYESIAN_INFERENCE.md#the-evidence-and-model-comparison));
on its own it has no meaning. The
reduced chi-square is the quadratic form of the noise model divided by the
degrees of freedom: close to 1 when the model and the noise model describe
the image, larger when the galaxy has structure the model lacks or the
noise model misses something.

## Images

| Attribute | Content |
| --- | --- |
| `fit.best_model_image` | the model at the best sample, on the cutout |
| `fit.residual` | data minus model |
| `fit.norm_residual` | the residual divided by the noise of each pixel; `NaN` on masked pixels |

`fit.summary()` collects the model name, the fit statistics, the evidence,
the best parameters and the noise model that was used.

## Correlated parameters

The flux, the effective radius and the Sérsic index are strongly correlated:
a brighter, larger, more concentrated profile can look much like a fainter,
smaller one. Read them together, with the corner plot of
[Plotting](PLOTTING.md), rather than one at a time. When you know the truth,
as in a simulation, `mahalanobis_distance` measures how far it lies from the
posterior while accounting for the correlations:

```python
truth = {"flux": 4.0e5, "re": 15.0, "n": 2.5}
posterior.mahalanobis_distance(truth)
# {'parameters': [...], 'distance_squared': ..., 'distance': ...,
#  'dof': 3, 'p_value': ...}
```

## Reading results later

Every fit writes its products to `fits/<model>/` in the run directory, and
the weighted posterior to `posterior.npz`. In a new session, load it with:

```python
from baymorph import InferenceResult

posterior = InferenceResult.load("runs/galaxy/fits/single_sersic/posterior.npz")
```

The images are NumPy files in the same directory, read with `numpy.load`,
and `fit_summary.json` holds the statistics and the evidence. Every file is
described in [Output Files](../reference/OUTPUT_FILES.md).
