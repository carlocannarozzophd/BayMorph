"""Utilities for exporting observation metadata."""

from __future__ import annotations

from typing import Any

from baymorph.core.observation import Observation


def observation_metadata_dict(observation: Observation) -> dict[str, Any]:
    """Convert an Observation into JSON-serializable metadata.

    Parameters
    ----------
    observation : Observation
        Observation providing normalized metadata.

    Returns
    -------
    dict
        Dictionary containing observation metadata suitable for JSON
        serialization.
    """
    metadata = observation.metadata
    image = getattr(observation, "image", None)

    return {
        "telescope": metadata.telescope,
        "instrument": metadata.instrument,
        "filter": metadata.filter_name,
        "pixel_scale_arcsec": metadata.pixel_scale,
        "seeing_fwhm": metadata.seeing_fwhm,
        "seeing_fwhm_unit": metadata.seeing_fwhm_unit,
        "seeing_fwhm_arcsec": (
            metadata.seeing_fwhm if metadata.seeing_fwhm_unit == "arcsec" else None
        ),
        "zeropoint": metadata.zeropoint,
        "gain": metadata.gain,
        "saturation": metadata.saturation,
        "exposure_time_s": metadata.exposure_time,
        "total_exposure_time_s": metadata.total_exposure_time,
        "background_header": metadata.background_header,
        "bunit": metadata.bunit,
        "hdu": (
            None
            if image is None
            else {"index": image.hdu_index, "name": image.hdu_name}
        ),
        "header_reading": (
            None
            if metadata.header_reading is None
            else metadata.header_reading.as_dict()
        ),
    }
