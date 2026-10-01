"""Exponential disc morphology."""

from __future__ import annotations

from baymorph.models.sersic import SersicModel


class ExponentialComponent(SersicModel):
    """Elliptical exponential surface-brightness component.

    Parameters
    ----------
    flux : float
        Total integrated flux over the continuous profile.
    re : float
        Effective half-light radius in pixels.
    q : float
        Projected axis ratio.
    theta : float
        Position angle in radians.
    x0, y0 : float
        Component centre in global pixel coordinates.

    Notes
    -----
    This component is exactly a Sérsic profile with index ``n = 1``.
    The effective radius is therefore the half-light radius, not the
    exponential scale length.
    """

    def __init__(
        self,
        *,
        flux: float,
        re: float,
        q: float,
        theta: float,
        x0: float,
        y0: float,
    ) -> None:
        super().__init__(
            flux=flux,
            re=re,
            n=1.0,
            q=q,
            theta=theta,
            x0=x0,
            y0=y0,
        )

    def scaled(self, factor: int) -> "ExponentialComponent":
        """Return this disc on a pixel grid finer by ``factor``.

        See :meth:`SersicModel.scaled`.
        """
        return ExponentialComponent(
            flux=self.flux,
            re=self.re * factor,
            q=self.q,
            theta=self.theta,
            x0=factor * (self.x0 + 0.5) - 0.5,
            y0=factor * (self.y0 + 0.5) - 0.5,
        )

    def render_components(
        self,
        shape: tuple[int, int],
        **kwargs,
    ):
        """Render the named exponential-disc contribution."""
        return {
            "exponential": self.render(
                shape=shape,
                **kwargs,
            )
        }
