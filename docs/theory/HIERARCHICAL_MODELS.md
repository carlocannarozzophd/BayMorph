# Hierarchical Models

When a galaxy is observed several times, in different bands or with
different instruments, its morphology is neither identical in every image
nor unrelated between them. This page explains how BayMorph fits several
images together and how a hierarchical model lets them share information.

## The joint likelihood

Each of the $K$ images $D_1, \dots, D_K$ keeps its own pixel grid, PSF,
mask, noise and residual background, and its own forward model with
parameters $\theta_k$. Assuming that the noise of different images is
independent, the likelihood of all the images is the product of their
likelihoods:

$$
L(D_1, \dots, D_K \mid \Theta) = \prod_{k=1}^{K} L_k(D_k \mid \theta_k),
$$

where $\Theta$ is the set of sampled parameters and the coupling of each
parameter decides how the $\theta_k$ follow from it. A single image is the
case $K = 1$.

## Sharing, independence and partial pooling

For each parameter there are three ways to relate the images:

- **Exact sharing**: one value for all the images, $\theta_k = \theta$, with
  one prior. Right when the quantity is the same in every image, such as
  the position of the galaxy on the sky.
- **Independence**: one value per image, each with its own prior. Right
  when the quantity differs freely, such as the flux in different bands.
- **Partial pooling**: one value per image, tied to the others by a
  dependence that the data constrain. The values may differ, but the images
  inform each other: a well-measured image pulls the estimate of a noisy one
  towards it when they agree. This is a hierarchical model, the structure
  used in astronomy to combine many uncertain measurements
  ([Hogg, Myers & Bovy 2010][hogg-myers-bovy2010]).

## The hierarchical coupling of BayMorph

BayMorph builds partial pooling with a Gaussian copula: the dependence
between the images is a correlation of Gaussian latent variables, and each
latent variable is mapped onto the parameter through its declared prior. A
copula separates the marginal distribution of each variable from the
dependence between them ([Scherrer et al. 2010][scherrer2010]). For a
parameter with prior $\pi$, of cumulative distribution $F$:

1. one independent standard-normal latent value $u_k$ per image is sampled;
2. a correlation $\rho \in [0, 1]$ is sampled, with prior
   $\mathrm{Beta}(2, 2)$;
3. the latent values are made equicorrelated,
   $$
   z_k = \sqrt{1 - \rho}\,(u_k - \bar{u}) + \sqrt{1 + (K - 1)\rho}\;\bar{u},
   $$
   so that every $z_k$ is a standard normal and every pair has correlation
   $\rho$;
4. the parameter of image $k$ is $\theta_k = F^{-1}\!\left(\Phi(z_k)\right)$,
   with $\Phi$ the standard normal cumulative distribution.

The construction is BayMorph's own. It has three properties:

- the value in each image has exactly the prior you declared, whatever
  $\rho$;
- $\rho = 0$ makes the images independent and $\rho \to 1$ makes them share
  one value;
- $\rho$ is inferred from the images: they decide how much they share.

**Wavelength coupling.** When the images are taken at different
wavelengths, the correlation can decrease with the distance in wavelength.
BayMorph then correlates the latent values of images $k$ and $l$ by

$$
R_{kl} = r_\star^{\,d_{kl}^2},
\qquad
d_{kl} = \frac{|\ln(\lambda_k / \lambda_l)|}{\ln 2},
$$

where $r_\star$, the correlation of two images a factor of two apart in
wavelength, is sampled with prior $\mathrm{Beta}(2, 2)$. Images at the same
wavelength are then fully correlated, and the correlation fades for widely
separated bands.

![Four panels: samples of the effective radius in two images under the hierarchical prior with rho of 0, 0.6 and 0.95, uniform in each image and increasingly concentrated along the diagonal, and the latent correlation of the wavelength coupling against the wavelength ratio for three reference correlations](figures/hierarchical_coupling-light.png#gh-light-mode-only)
![Four panels: samples of the effective radius in two images under the hierarchical prior with rho of 0, 0.6 and 0.95, uniform in each image and increasingly concentrated along the diagonal, and the latent correlation of the wavelength coupling against the wavelength ratio for three reference correlations](figures/hierarchical_coupling-dark.png#gh-dark-mode-only)

*The prior of a hierarchical effective radius, uniform between 0.5 and 5 arcsec, in two images, drawn with the coupling of BayMorph for three values of ρ: each image keeps its uniform prior, and the values concentrate along the diagonal as ρ grows. Bottom right: the latent correlation of the wavelength coupling, which equals $r_\star$ at a factor of two in wavelength.*

**In the results.** The posterior holds the sampled coordinates of a
hierarchical parameter: its correlation, `<name>__rho`, or for wavelength
coupling `<name>__rstar`, and one latent value per image,
`<name>__z__<image>`. The value of the parameter in each image is a
function of them; at the best sample it is in `result.observations`. The
configuration is described in
[Several Observations](../guide/MULTI_OBSERVATION.md).

## Comparing couplings

Whether two images share a radius, or only partially, is itself a question
for the evidence: fit the same images with `shared`, `hierarchical` and
`independent` coupling and compare the evidences, with the same images,
masks and noise models, and priors fixed in advance
([Trotta 2008][trotta2008]). The hierarchical coupling preserves the prior
of each image, so the three models differ only in how the images are tied.

## Limits

- The noise of different images is assumed independent: a correlation of
  the residuals between images, as in images that share exposures, is not
  modelled.
- A coupling is always declared: BayMorph never infers it from the names of
  the images.
- The six models with bars, rings, spiral arms, broken discs, boxy isophotes
  or cores are not available for joint fits.

## References

- [Hogg D. W., Myers A. D., Bovy J., 2010, ApJ, 725, 2166][hogg-myers-bovy2010]: hierarchical inference from many uncertain measurements.
- [Scherrer R. J. et al., 2010, ApJ, 708, L9][scherrer2010]: the copula, and the Gaussian copula, in astronomy.
- [Trotta R., 2008, Contemporary Physics, 49, 71][trotta2008]: model comparison with the evidence.

[hogg-myers-bovy2010]: https://ui.adsabs.harvard.edu/abs/2010ApJ...725.2166H/abstract
[scherrer2010]: https://ui.adsabs.harvard.edu/abs/2010ApJ...708L...9S/abstract
[trotta2008]: https://ui.adsabs.harvard.edu/abs/2008ConPh..49...71T/abstract
