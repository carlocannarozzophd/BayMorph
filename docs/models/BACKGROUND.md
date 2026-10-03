# Sky and Background

The sky is handled in two steps. Preparation measures the sky around the
galaxy and subtracts it as a constant; the fit then infers what is left,
the *residual background*, together with the galaxy. The residual
background belongs to the observation: it is not part of the galaxy, and no
morphology model contains it.

## The sky in preparation

`prepare_image()` measures the elliptical radial profile of the target,
the median of the pixels in elliptical annuli, with the other sources
masked, and finds its *plateau*: the radius beyond which the profile stops
falling and stays flat within the noise. Everything beyond the plateau
radius plus 5 pixels, minus the masked sources, is the *sky region*. There
BayMorph

1. clips the outliers and subtracts the median of what remains from the
   cutout, as a constant;
2. measures the noise scale of the sky, about a plane fitted to the sky
   region, so that a gradient is not mistaken for noise; the scale corrects
   for the truncation of the clipping;
3. reports the gradient of that plane, with its significance, as a
   diagnostic (`image.sky_gradient`).

![Two panels: a galaxy with a masked neighbour and the sky region shaded beyond an ellipse, and its radial profile falling to a flat sky level, with the plateau radius marked and the sky region shaded](figures/sky_region-light.png#gh-light-mode-only)
![Two panels: a galaxy with a masked neighbour and the sky region shaded beyond an ellipse, and its radial profile falling to a flat sky level, with the plateau radius marked and the sky region shaded](figures/sky_region-dark.png#gh-dark-mode-only)

*A simulated galaxy on a sky of 100 image units, prepared with the functions of BayMorph. Left: the plateau ellipse (solid), the inner edge of the sky region 5 px beyond it (dashed), the sky region (shaded) and a masked neighbour. Right: the radial profile, which reaches the sky level at the plateau radius.*

The faint outskirts of a galaxy are where the sky matters: Sérsic fits are
reliable when the effective surface brightness of the galaxy is above that
of the sky ([Häussler et al. 2007][haussler2007]). A sky estimated too close
to a galaxy contains part of its light, and subtracting it makes the
profile fall faster, which lowers the fitted flux, radius and index. The
residual background absorbs a constant error of the subtracted level, but
not light of the galaxy that falls off with radius, so the cutout must
reach well into the sky: see [Image and Cutout](../guide/IMAGE.md).

## The residual background

`fit.background.model` selects one of two forms.

### Constant (`constant`)

$$
B(x, y) = b.
$$

| Parameter | Meaning | Units | Default prior |
| --- | --- | --- | --- |
| `background_level` | residual background level | image units | uniform −50 to 50 |

The default prior is not scaled to the image: choose it from the noise and
the units of each image.

### Plane (`plane`)

$$
B(x, y) = b + g_x\,(x - x_{\mathrm{ref}}) + g_y\,(y - y_{\mathrm{ref}}),
\qquad
x_{\mathrm{ref}} = \frac{n_x - 1}{2},
\quad
y_{\mathrm{ref}} = \frac{n_y - 1}{2},
$$

where $n_x$ and $n_y$ are the sides of the cutout, so that `background_level`
is the level at its geometric centre. It is the sky of GALFIT
([Peng et al. 2010][peng2010]): a level at a reference point and two
gradients.

| Parameter | Meaning | Units | Default prior |
| --- | --- | --- | --- |
| `background_level` | level at the centre of the cutout | image units | uniform −50 to 50 |
| `background_gradient_x` | gradient along increasing $x$ | image units per pixel | uniform −0.1 to 0.1 |
| `background_gradient_y` | gradient along increasing $y$ | image units per pixel | uniform −0.1 to 0.1 |

![Two images on a blue-to-red scale: a constant residual background and a tilted plane, with its reference point at the centre of the cutout and an arrow along the gradient](figures/background_models-light.png#gh-light-mode-only)
![Two images on a blue-to-red scale: a constant residual background and a tilted plane, with its reference point at the centre of the cutout and an arrow along the gradient](figures/background_models-dark.png#gh-dark-mode-only)

*A constant background and a plane with the same level b at the centre of the cutout; the arrow shows the direction of the gradient.*

**When to use a plane.** When the sky gradient that preparation reports is
significant at several sigma. The choice remains yours: BayMorph reports the
gradient and never switches the model.

## In joint fits

The background model is declared once, in the joint `fit.background`
section, and its parameters are declared under
`fit.multi_observation.parameters` with `independent` coupling, because each
image has its own residual background.

## Known degeneracies and limits

- The background is anticorrelated with the flux, the radius and the index
  of the galaxy, which add light far from the centre as it does.
- The sky region may still contain faint light of the galaxy, which biases
  the subtracted sky; how much, and how best to correct it, is being
  measured.
- Structure of the sky beyond a plane is not modelled.

## Validation

A controlled test fits a simulated image with a known sky gradient and
recovers the gradient, removing the residual structure that a constant
background leaves.

## Python

`baymorph.observation.background.ConstantBackground(level)` and
`TiltedPlaneBackground(level, gradient_x, gradient_y)`.

## References

- [Häussler B. et al., 2007, ApJS, 172, 615][haussler2007]: the reliability of Sérsic fits against the brightness of the sky.
- [Peng C. Y. et al., 2010, AJ, 139, 2097][peng2010]: the tilted-plane sky.

[haussler2007]: https://ui.adsabs.harvard.edu/abs/2007ApJS..172..615H/abstract
[peng2010]: https://ui.adsabs.harvard.edu/abs/2010AJ....139.2097P/abstract
