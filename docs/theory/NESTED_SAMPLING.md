# Nested Sampling

A fit needs both the posterior and the evidence. Nested sampling computes
the evidence directly, and returns the posterior as a by-product
([Skilling 2004][skilling2004]); it is the method of the default sampler of
BayMorph, Nautilus.

## The evidence as a one-dimensional integral

Call $X(\lambda)$ the fraction of the prior volume where the likelihood
exceeds $\lambda$: $X = 1$ for the lowest likelihood and $X \to 0$ at its
maximum. The evidence, an integral over all the parameters, becomes an
integral over that single variable:

$$
Z = \int L(\theta)\,\pi(\theta)\,d\theta = \int_0^1 L(X)\,dX,
\qquad
X(\lambda) = \int_{L(\theta) > \lambda} \pi(\theta)\,d\theta .
$$

Nested sampling estimates $L(X)$ by climbing the likelihood while keeping
track of how much prior volume is left.

## The algorithm

1. Draw $n$ *live points* from the prior.
2. Remove the live point of lowest likelihood, $L_i$, and replace it with a
   new point drawn from the prior with a higher likelihood.
3. Each step shrinks the prior volume enclosed by the live points by a factor
   of about $e^{-1/n}$, so after $i$ steps $\ln X_i \approx -i/n$.
4. Repeat until the live points can no longer change the evidence
   appreciably.

The evidence and the posterior follow from the discarded points:

$$
Z \approx \sum_i L_i\,(X_{i-1} - X_i),
\qquad
p_i = \frac{L_i\,(X_{i-1} - X_i)}{Z},
$$

where $p_i$ is the posterior weight of the $i$-th discarded point. Because
the shrinkage of each step is random, $\ln Z$ has an uncertainty of about
$\sqrt{H/n}$, where $H = \sum_i p_i \ln(L_i/Z)$ measures how much the image
narrows the prior. More live points make the evidence more precise and
explore several posterior modes more reliably, at a higher cost.

![Four panels from a small nested sampling run on a two-dimensional Gaussian likelihood: the discarded points converging on the peak, the log-likelihood against the log of the prior volume, the posterior weights peaking at an intermediate prior volume, and the running log-evidence approaching the exact value](figures/nested_sampling-light.png#gh-light-mode-only)
![Four panels from a small nested sampling run on a two-dimensional Gaussian likelihood: the discarded points converging on the peak, the log-likelihood against the log of the prior volume, the posterior weights peaking at an intermediate prior volume, and the running log-evidence approaching the exact value](figures/nested_sampling-dark.png#gh-dark-mode-only)

*A run with 300 live points on a circular Gaussian likelihood of width 0.5 in a uniform prior of side 10. Top: the discarded points, the later the stronger their colour, close in on the peak, while the likelihood rises as the prior volume shrinks. Bottom left: the posterior mass lies where the likelihood is high but the volume not yet negligible. Bottom right: the running evidence reaches the exact value within its uncertainty, shaded. The figure uses an ideal sampler written for it; real samplers have to construct the region of higher likelihood.*

The hard part is step 2: drawing a new point from the prior inside the
region of higher likelihood, which in many dimensions is small and of
irregular shape. The samplers differ in how they do it.

## The samplers of BayMorph

| Sampler | How it draws new points |
| --- | --- |
| [Nautilus](https://nautilus-sampler.readthedocs.io/) (default) | Importance nested sampling: neural networks learn the shape of the likelihood contours and propose points inside them, and every likelihood evaluation contributes to the posterior and the evidence ([Lange 2023][lange2023]). |
| [Dynesty](https://dynesty.readthedocs.io/) | Bounding ellipsoids and random walks or slices within them ([Speagle 2020][speagle2020]); in dynamic mode, it adds live points where the posterior mass lies ([Higson et al. 2019][higson2019]). |
| [pocoMC](https://pocomc.readthedocs.io/) | Not nested sampling but preconditioned Monte Carlo: a sequence of distributions from the prior to the posterior, sampled by sequential Monte Carlo after a normalizing flow has removed the correlations between the parameters; it also estimates the evidence ([Karamanis et al. 2022][karamanis2022]). |

All three return weighted posterior samples and an evidence, which BayMorph
stores in the same result object. The choice of sampler changes how the
posterior is computed, never the scientific problem.

## In practice

- **The weights are the result.** Summaries, intervals and evidences are
  computed from the weighted samples; the equal-weight samples BayMorph
  derives from them are for plots.
- **Live points and stopping.** For Nautilus, `n_live` sets the number of
  live points, 2000 by default, and `n_eff` the effective posterior sample
  size at which the run stops, 10000 by default. A quick look with fewer live
  points is fine to check a configuration; a result to publish deserves the
  defaults or more.
- **The uncertainty of the evidence** is reported as
  `posterior.logz_error`. A difference of log-evidences much smaller than
  that uncertainty says nothing.
- **Several modes.** A posterior with separate modes, such as the two
  exchangeable components of `double_sersic`, needs enough live points to
  find all of them. Repeating a run with another `seed` is the simplest
  check that a result is stable.
- **Interrupted runs** resume from a checkpoint with
  `fit.sampler.init.filepath: auto` for Nautilus; see
  [The Runner](../guide/RUNNER.md#running-again).

## References

- [Higson E. et al., 2019, Statistics and Computing, 29, 891][higson2019]: dynamic nested sampling.
- [Karamanis M. et al., 2022, MNRAS, 516, 1644][karamanis2022]: preconditioned Monte Carlo and pocoMC.
- [Lange J. U., 2023, MNRAS, 525, 3181][lange2023]: importance nested sampling with neural networks and Nautilus.
- [Skilling J., 2004, AIP Conf. Proc., 735, 395][skilling2004]: nested sampling.
- [Speagle J. S., 2020, MNRAS, 493, 3132][speagle2020]: Dynesty.

[higson2019]: https://ui.adsabs.harvard.edu/abs/2019S%26C....29..891H/abstract
[karamanis2022]: https://ui.adsabs.harvard.edu/abs/2022MNRAS.516.1644K/abstract
[lange2023]: https://ui.adsabs.harvard.edu/abs/2023MNRAS.525.3181L/abstract
[skilling2004]: https://ui.adsabs.harvard.edu/abs/2004AIPC..735..395S/abstract
[speagle2020]: https://ui.adsabs.harvard.edu/abs/2020MNRAS.493.3132S/abstract
