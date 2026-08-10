"""Observation transforms driven by declarative model geometry."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS

from baymorph.models.geometry import ModelGeometry
from baymorph.observation.geometry import sky_ellipse_to_detector


@dataclass(frozen=True)
class SkyModelParameterTransform:
    """Transform sky-frame model geometry to one detector grid.

    Parameters
    ----------
    wcs : astropy.wcs.WCS
        Celestial WCS of the target observation.
    geometry : ModelGeometry
        Declarative geometry contract of the production model.
    step_arcsec : float, default=0.01
        Finite-difference step used by the local WCS Jacobian.

    Notes
    -----
    The sky-frame source centre is represented by ``ra`` and ``dec`` in
    degrees.

    Each elliptical structure belongs to the same galaxy representation.
    Its sky-frame radius and position angle are transformed independently
    into the detector frame while the galaxy centre is transformed once.

    Non-geometric parameters are passed through unchanged.
    """

    wcs: WCS
    geometry: ModelGeometry
    step_arcsec: float = 0.01

    def __call__(
        self,
        parameters: Mapping[str, float],
    ) -> dict[str, float]:
        """Return detector-frame model parameters."""
        centre = SkyCoord(
            float(parameters["ra"]) * u.deg,
            float(parameters["dec"]) * u.deg,
            frame="icrs",
        )

        transformed = {
            name: float(value)
            for name, value in parameters.items()
            if name not in self._sky_geometry_names()
        }

        detector_centre = None

        for structure in self.geometry.elliptical_structures:
            radius_name = self.sky_radius_name(
                structure.radius
            )
            angle_name = self.sky_angle_name(
                structure.position_angle
            )

            detector = sky_ellipse_to_detector(
                wcs=self.wcs,
                centre=centre,
                re_arcsec=float(
                    parameters[radius_name]
                ),
                q=float(
                    parameters[
                        structure.axis_ratio
                    ]
                ),
                position_angle=float(
                    parameters[angle_name]
                ),
                step_arcsec=self.step_arcsec,
            )

            transformed[
                structure.radius
            ] = detector.re
            transformed[
                structure.axis_ratio
            ] = detector.q
            transformed[
                structure.position_angle
            ] = detector.theta

            if detector_centre is None:
                detector_centre = (
                    detector.x0,
                    detector.y0,
                )

        if detector_centre is None:
            x0, y0 = self.wcs.world_to_pixel(
                centre
            )
            detector_centre = (
                float(x0),
                float(y0),
            )

        transformed[
            self.geometry.centre_x
        ] = detector_centre[0]
        transformed[
            self.geometry.centre_y
        ] = detector_centre[1]

        return transformed

    def _sky_geometry_names(
        self,
    ) -> set[str]:
        names = {
            "ra",
            "dec",
        }

        for structure in self.geometry.elliptical_structures:
            names.add(
                self.sky_radius_name(
                    structure.radius
                )
            )
            names.add(
                structure.axis_ratio
            )
            names.add(
                self.sky_angle_name(
                    structure.position_angle
                )
            )

        return names

    @staticmethod
    def sky_radius_name(
        detector_name: str,
    ) -> str:
        """Return the sky-frame radius parameter name."""
        if detector_name == "re":
            return "re_arcsec"

        if detector_name.startswith("re_"):
            return (
                "re_arcsec_"
                + detector_name.removeprefix(
                    "re_"
                )
            )

        raise ValueError(
            "Elliptical radius parameters must use "
            "'re' or 're_<structure>' naming."
        )

    @staticmethod
    def sky_angle_name(
        detector_name: str,
    ) -> str:
        """Return the sky-frame position-angle parameter name."""
        if detector_name == "theta":
            return "position_angle"

        if detector_name.startswith("theta_"):
            return (
                "position_angle_"
                + detector_name.removeprefix(
                    "theta_"
                )
            )

        raise ValueError(
            "Elliptical angle parameters must use "
            "'theta' or 'theta_<structure>' naming."
        )
