# Residual Background

Preparation subtracts a constant sky from the image: the median of the
clipped sky region. Whatever background remains is a residual, inferred with
the morphology as an observation effect (AD-031). It is not part of the
galaxy, and the morphology models never contain it. `fit.background.model`
selects one of two forms.

## Constant (`constant`)

$$
B(x, y) = b.
$$

| Parameter | Meaning | Units | Default prior |
| --- | --- | --- | --- |
| `background_level` | residual background level | image units | uniform −50 to 50 |

The default prior is not scaled to the image: choose it from the noise and
the units of each image.

## Plane (`plane`)

$$
B(x, y) = b + g_x\,(x - x_{\mathrm{ref}}) + g_y\,(y - y_{\mathrm{ref}}),
\qquad
x_{\mathrm{ref}} = \frac{n_x - 1}{2},
\quad
y_{\mathrm{ref}} = \frac{n_y - 1}{2},
$$

where $n_x$ and $n_y$ are the sides of the cutout, so that `background_level`
is the level at its geometric centre.

| Parameter | Meaning | Units | Default prior |
| --- | --- | --- | --- |
| `background_level` | level at the centre of the cutout | image units | uniform −50 to 50 |
| `background_gradient_x` | gradient along increasing $x$ | image units per pixel | uniform −0.1 to 0.1 |
| `background_gradient_y` | gradient along increasing $y$ | image units per pixel | uniform −0.1 to 0.1 |

![Two images on a blue-to-red scale: a constant residual background and a tilted plane, with its reference point at the centre of the cutout and an arrow along the gradient](figures/background_models-light.png#gh-light-mode-only)
![Two images on a blue-to-red scale: a constant residual background and a tilted plane, with its reference point at the centre of the cutout and an arrow along the gradient](figures/background_models-dark.png#gh-dark-mode-only)

*A constant background and a plane with the same level b at the centre of the cutout; the arrow shows the direction of the gradient.*

**When to use it.** Preparation measures the noise scale about a plane
fitted to the sky region and reports the fitted gradient and its
significance (AD-041). A significant gradient indicates that a plane is
warranted; the choice remains the user's, and BayMorph never makes it.

## In joint fits

The background family is declared once, in the joint `fit.background`
section, and its parameters are declared under
`fit.multi_observation.parameters` with `independent` coupling, because each
observation has its own residual background (AD-044).

## Known degeneracies and limits

- The background is anticorrelated with the flux, the radius and the index
  of the galaxy, which add light far from the centre as it does.
- The sky region can contain the outer light of the galaxy. On the first
  real target, the target still contributed about 0.15 of the noise scale
  beyond the plateau radius
  ([noise and background chain](../research/NOISE_BACKGROUND_AUDIT_2026-10-01.md)),
  which biases the subtracted sky and therefore the residual background. The
  background is the next step of the revision of the observation model.
- Structure of the sky beyond a plane is not modelled.

## Validation

A controlled test fits a simulated image with a known sky gradient and
recovers the gradient, removing the residual structure that a constant
background leaves.

## Python

`baymorph.observation.background.ConstantBackground(level)` and
`TiltedPlaneBackground(level, gradient_x, gradient_y)`.
