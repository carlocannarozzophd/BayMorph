# Sky and Noise

## The sky

During `prepare_image()`, BayMorph finds the radius where the radial
profile of the galaxy flattens into the sky, the plateau. Everything beyond
that radius, minus the masked sources, is the sky region. The median of the
sky region, after clipping outliers, is subtracted from the cutout as a
constant.

Whatever background is left, because the sky is not perfectly flat or the
median is not exactly its level, is fitted together with the galaxy as the
*residual background*:

=== "Constant"

    ```yaml
    fit:
      background:
        model: constant
        priors:
          background_level: {type: uniform, lower: -20, upper: 20}
    ```

    One parameter, `background_level`, in image units.

=== "Plane"

    ```yaml
    fit:
      background:
        model: plane
        priors:
          background_level: {type: uniform, lower: -20, upper: 20}
          background_gradient_x: {type: uniform, lower: -0.05, upper: 0.05}
          background_gradient_y: {type: uniform, lower: -0.05, upper: 0.05}
    ```

    A level at the centre of the cutout and two gradients, in image units
    per pixel.

The default prior of `background_level`, −50 to 50, is not scaled to your
image: choose a range of a few times the noise scale. The
[Sky and Background](../models/BACKGROUND.md) page defines both models.

### Is the sky tilted?

Preparation fits a plane to the sky region and reports its gradient, with
its significance, as a diagnostic:

```python
image = runner.prepare_image()

for axis in ("x", "y"):
    value = image.sky_gradient[f"gradient_{axis}"]
    significance = image.sky_gradient[f"gradient_{axis}_significance"]
    print(f"gradient_{axis}: {value:.2g} per pixel ({significance:.1f} sigma)")
```

A gradient significant at several sigma suggests `model: plane`. BayMorph
reports it but never switches the model for you. `sky_gradient` is `None`
when no gradient could be measured.

## The noise

BayMorph measures the noise scale of the sky on the sky region of the
cutout, after removing a plane, so that a gradient is not mistaken for noise:

```python
import numpy as np

print(f"Sky subtracted: {image.background:.1f}")
print(f"Sky noise scale: {np.median(image.noise):.2f}")
```

On the example image these print 400.5 and 20.60, as expected for a sky of
400 counts with Poisson noise and a read noise of 5 counts.

The likelihood then describes the noise of every pixel with three parts:

1. **the noise of the sky**, the scale measured above;
2. **the Poisson noise of the galaxy**, which grows with its brightness: the
   model of the galaxy divided by the *effective gain*;
3. **the correlation of the sky** between nearby pixels, which resampled
   or coadded images usually have, measured on the sky region before the
   fit.

[Noise Models](../models/NOISE.md) gives the formula, and the
[Configuration reference](../reference/CONFIGURATION.md#likelihood) its
options.

## The effective gain

The effective gain is the number of detected photons, or electrons, per
image unit: the variance of the Poisson noise of the galaxy is the model
divided by the gain. BayMorph never fits it, and has no default for it:

| `effective_gain.source` | Use it when | Example |
| --- | --- | --- |
| `declared` | you know the gain | an image in electrons: `gain: 1`; in ADU of a detector with 2.5 electrons per ADU: `gain: 2.5`; in electrons per second of a 600 s exposure: `gain: 600` |
| `measured_sky` | the sky is still in the image | BayMorph divides the sky level by the square of the sky noise |
| `sky_level` | the sky was subtracted before, but you know its level, for example from the header | `sky_level: 1250.0` |
| `absent` | the image has no Poisson noise from the source, as some simulations | |

```yaml
fit:
  likelihood:
    type: correlated_gaussian
    effective_gain:
      source: declared
      gain: 2.5
```

`measured_sky` and `sky_level` assume that the sky noise is the Poisson
noise of the sky. They require a sky level at least ten times the sky noise,
and they return a slightly lower gain when read noise is not negligible: on
the example image, 0.944 instead of 1. `baymorph config` and
`make_config_from_fits` write `measured_sky` when the sky is still in the
image, and otherwise leave the gain for you to declare: the fit refuses to
start without it.

The gain used by a fit is recorded in its summary:

```python
fit.summary()["likelihood"]["effective_gain"]
```

## Limits

- The sky noise, its correlation and the gain are measured or declared
  once, before the fit. Their uncertainty is not propagated into the
  posterior.
- An external weight or variance map is not yet supported:
  `data.noise_map_path` is rejected.
- The correlation of the sky is estimated from the sky region of the
  cutout; a small cutout gives a noisy estimate.
