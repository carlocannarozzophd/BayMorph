"""Observation-specific transformations of scientific model parameters."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS

from baymorph.observation.geometry import (
    sky_ellipse_to_detector,
)


@dataclass(frozen=True)
class SkyEllipseParameterTransform:
    """Transform sky-frame elliptical morphology to one detector grid.

    Parameters
    ----------
    wcs : astropy.wcs.WCS
        Celestial WCS of the target observation.
    step_arcsec : float, default=0.01
        Finite-difference step used by the local WCS Jacobian.

    Notes
    -----
    The input parameter mapping must contain ``ra``, ``dec``,
    ``re_arcsec``, ``q``, and ``position_angle``.

    ``ra`` and ``dec`` are interpreted in degrees. ``re_arcsec`` is the
    sky-frame effective semi-major-axis radius in arcseconds.
    ``position_angle`` follows the local east-to-north tangent-plane
    convention used by the BayMorph geometry layer.

    Parameters unrelated to elliptical geometry are preserved unchanged.
    """

    wcs: WCS
    step_arcsec: float = 0.01

    def __call__(
        self,
        parameters: Mapping[str, float],
    ) -> dict[str, float]:
        """Return detector-frame representation parameters."""
        required = {
            "ra",
            "dec",
            "re_arcsec",
            "q",
            "position_angle",
        }

        missing = required.difference(
            parameters
        )

        if missing:
            names = ", ".join(
                sorted(missing)
            )
            raise KeyError(
                "Sky-frame ellipse parameters are missing: "
                f"{names}."
            )

        centre = SkyCoord(
            float(parameters["ra"]) * u.deg,
            float(parameters["dec"]) * u.deg,
            frame="icrs",
        )

        geometry = sky_ellipse_to_detector(
            wcs=self.wcs,
            centre=centre,
            re_arcsec=float(
                parameters["re_arcsec"]
            ),
            q=float(
                parameters["q"]
            ),
            position_angle=float(
                parameters["position_angle"]
            ),
            step_arcsec=self.step_arcsec,
        )

        transformed = {
            name: float(value)
            for name, value in parameters.items()
            if name not in required
        }

        transformed.update(
            {
                "x0": geometry.x0,
                "y0": geometry.y0,
                "re": geometry.re,
                "q": geometry.q,
                "theta": geometry.theta,
            }
        )

        return transformed
