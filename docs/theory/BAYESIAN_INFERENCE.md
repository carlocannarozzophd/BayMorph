# Bayesian Inference

BayMorph treats the morphology of a galaxy as an inference problem: given an
image, which values of the parameters of a model are probable, and how well
does the model explain the image compared with another? This page explains
the quantities a fit returns and how to read them.

## Bayes' theorem for an image

For an image $D$ and a model $M$ with parameters $\theta$, such as the flux,
the effective radius and the Sérsic index,

$$
p(\theta \mid D, M) =
\frac{L(D \mid \theta, M)\;\pi(\theta \mid M)}{Z(D \mid M)},
\qquad
Z(D \mid M) = \int L(D \mid \theta, M)\,\pi(\theta \mid M)\,d\theta .
$$

- The **likelihood** $L$ is the probability of the image if the parameters
  were $\theta$. BayMorph computes it in two steps: a forward model turns
  $\theta$ into the expected image, $\mu = P \circledast I(\theta) + B$
  (see [Models](../models/index.md)), and the
  [noise model](../models/NOISE.md) gives the probability of the observed
  pixels around it. Writing down such a generative model of the data is
  what makes a fit well defined
  ([Hogg, Bovy & Lang 2010][hogg-bovy-lang2010]).
- The **prior** $\pi$ states what is known before looking at the image. In
  BayMorph every sampled parameter has an explicit prior, declared in the
  configuration and recorded with the results.
- The **posterior** $p(\theta \mid D, M)$ is the answer: the probability of
  every combination of parameter values, given the image.
- The **evidence** $Z$ is the probability of the image under the model as a
  whole, averaged over its prior. It normalizes the posterior and compares
  models.

## Reading the posterior

A fit returns the posterior as weighted samples (see
[Nested Sampling](NESTED_SAMPLING.md)). The marginal posterior of one
parameter is the weighted histogram of its samples, summarized by its median
and its 16th and 84th percentiles, the 68 percent credible interval of
`posterior.summary(name)`.

The parameters of a galaxy model are rarely independent. A more
concentrated profile, with a larger Sérsic index, puts more light both in
its centre and in its wings, and a larger radius can compensate part of
that, so the radius and the index are correlated:

![Two panels: the likelihood of the effective radius and the Sérsic index of a simulated galaxy, an elongated diagonal region around the true values, and a likelihood under a narrow and a wide prior, with the area under each product giving the evidence](figures/likelihood_and_occam-light.png#gh-light-mode-only)
![Two panels: the likelihood of the effective radius and the Sérsic index of a simulated galaxy, an elongated diagonal region around the true values, and a likelihood under a narrow and a wide prior, with the area under each product giving the evidence](figures/likelihood_and_occam-dark.png#gh-dark-mode-only)

*Left: the likelihood of $R_e$ and $n$ for a simulated Sérsic galaxy, computed with BayMorph, with the flux at its best value for each pair; the contours enclose 68, 95 and 99.7 percent. Right: the same likelihood under a narrow and a wide uniform prior; the shaded areas are the evidences.*

The marginal interval of one parameter already includes the uncertainty of
all the others. Two correlated parameters should be read together, in the
corner plot, rather than one at a time.

What the posterior does **not** include:

- the uncertainty of the inputs fixed before the fit: the PSF, the
  subtracted sky, the sky noise, its correlation and the effective gain;
- the error of a model that cannot describe the galaxy: a single Sérsic
  fitted to a barred galaxy gives a narrow posterior around values that
  describe neither the bar nor the disc. The residuals, not the posterior
  widths, reveal it.

## The evidence and model comparison

The evidence averages the likelihood over the prior. A model whose prior
spreads over values the image rules out pays for it, because most of its
prior volume has a low likelihood: the evidence embodies Occam's razor. In
the figure above, the same likelihood gives an evidence three times smaller
under a prior three times wider.

Two models, $M_1$ and $M_2$, fitted to the same image are compared by the
ratio of their evidences, the Bayes factor, usually through its logarithm:

$$
\ln B_{12} = \ln Z_1 - \ln Z_2 .
$$

A common reading of $|\ln B_{12}|$ is: below 1, inconclusive; from 1, weak;
from 2.5, moderate; above 5, strong evidence for the model with the larger
evidence ([Trotta 2008][trotta2008]). The comparison is meaningful only if
the two fits share

- the same prepared image: the same cutout, mask and sky subtraction;
- the same noise model, with the same inputs;
- priors chosen before looking at the results.

BayMorph requires every likelihood to be completely normalized, so its
evidences are absolute. Evidences obtained with different masks or cutouts
are not comparable, and a mask chosen after looking at the residuals of a
fit makes the comparison depend on that choice. Because the evidence
depends on the prior volume, a prior must be reported with every evidence.

## Checking a fit

The posterior answers the question within the model; it does not check the
model. Check every fit with

- the residuals and the normalized residuals: structure in them is light the
  model does not describe;
- the reduced chi-square: close to 1 when the model and the noise model
  describe the image;
- the radial profile with its posterior bands;
- simulations: fitting images made with known parameters, many times with
  different noise, is the only way to know whether the intervals cover the
  truth as often as they claim.

## References

- [Hogg D. W., Bovy J., Lang D., 2010, arXiv:1008.4686][hogg-bovy-lang2010]: fitting a model to data with a generative model.
- [Trotta R., 2008, Contemporary Physics, 49, 71][trotta2008]: Bayesian inference and model selection, with the scale of the Bayes factor.

[hogg-bovy-lang2010]: https://ui.adsabs.harvard.edu/abs/2010arXiv1008.4686H/abstract
[trotta2008]: https://ui.adsabs.harvard.edu/abs/2008ConPh..49...71T/abstract
