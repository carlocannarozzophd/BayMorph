import pytest

import numpy as np

from astropy.io import fits

from baymorph.pipeline import observation_metadata_dict

from baymorph.core.image import ImageData
from baymorph.core.observation import Observation


def test_observation_metadata_dict():
    header = fits.Header()

    header["FILTER"] = "i_SDSS"
    header["TELESCOP"] = "ESO-VST"
    header["INSTRUME"] = "OMEGACAM"
    header["PHOTZP"] = 30.0
    header["GAIN"] = 2.5
    header["SATLEVEL"] = 55000.0
    header["FWHM"] = 0.72
    header["EXPTIME"] = 1200.0
    header["TEXPTIME"] = 3600.0
    header["BKGMED"] = 12.3
    header["BUNIT"] = "ADU"

    header["CTYPE1"] = "RA---TAN"
    header["CTYPE2"] = "DEC--TAN"
    header["CRPIX1"] = 16.0
    header["CRPIX2"] = 11.0
    header["CRVAL1"] = 150.0
    header["CRVAL2"] = 2.0
    header["CDELT1"] = -0.2 / 3600.0
    header["CDELT2"] = 0.2 / 3600.0

    image = ImageData(
        data=np.zeros((21, 31)),
        header=header,
    )

    observation = Observation.from_image(image)

    metadata = observation_metadata_dict(
        observation,
    )

    assert metadata["telescope"] == "ESO-VST"
    assert metadata["instrument"] == "OMEGACAM"
    assert metadata["filter"] == "i_SDSS"

    assert metadata["pixel_scale_arcsec"] == pytest.approx(0.2)

    assert metadata["seeing_fwhm"] == 0.72

    assert metadata["seeing_fwhm_unit"] is None

    assert metadata["seeing_fwhm_arcsec"] is None

    assert metadata["zeropoint"] == 30.0

    assert metadata["gain"] == 2.5

    assert metadata["saturation"] == 55000.0

    assert metadata["exposure_time_s"] == 1200.0

    assert metadata["total_exposure_time_s"] == 3600.0

    assert metadata["background_header"] == 12.3

    assert metadata["bunit"] == "ADU"


def test_explicit_arcsec_fwhm_is_persisted_safely() -> None:
    """The legacy arcsec field requires an explicit arcsec declaration."""
    from types import SimpleNamespace

    from baymorph.core.observation import ObservationMetadata
    from baymorph.pipeline.observation_metadata import (
        observation_metadata_dict,
    )

    observation = SimpleNamespace(
        metadata=ObservationMetadata(
            width=11,
            height=9,
            center_x=5.0,
            center_y=4.0,
            seeing_fwhm=0.72,
            seeing_fwhm_unit="arcsec",
        )
    )

    metadata = observation_metadata_dict(
        observation
    )

    assert metadata["seeing_fwhm"] == 0.72
    assert metadata["seeing_fwhm_unit"] == "arcsec"
    assert metadata["seeing_fwhm_arcsec"] == 0.72
