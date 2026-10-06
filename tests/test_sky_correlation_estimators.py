"""The sky-correlation estimators of the noise model and their declaration.

The estimator ``autocovariance`` assumes that the correlation of the sky ends
within the kernel. A sky that also holds a large-scale field breaks that
assumption: its kernel is inflated or rejected. The estimator
``field_separated`` fits the field beyond the kernel and removes it. The
images below are a stationary noise with a separable AR(1) correlation, alone
or with a Gaussian random field of exponential correlation, on a ring of sky
around a masked galaxy.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.signal import lfilter

import baymorph.pipeline.noise_calibration as noise_calibration
from baymorph.core.correlated_noise import (
    CALIBRATION_SCHEMA_VERSION,
    CORRELATION_ESTIMATOR,
    CORRELATION_MAX_LAG,
    FIELD_LENGTH_BOUNDS,
    FIELD_SEPARATED_ESTIMATOR,
    FIELD_SEPARATED_GROWTH_PIXELS,
    calibrate_correlated_gaussian_noise,
)
from baymorph.pipeline.noise_calibration import (
    build_correlated_noise_calibration,
    configured_sky_correlation,
    load_correlated_noise_calibration,
    save_correlated_noise_calibration,
    validate_correlated_noise_calibration_estimator,
)

SHAPE = (400, 400)
SEEDS = range(6)
NOISE_CORRELATION = 0.3
FIELD_LENGTH = 40.0
FIELD_VARIANCE = 0.07
"""A field of 7 percent of the variance of the noise, as measured on the
r-band sky of the target, on a length of tens of pixels."""


def _noise(seed: int) -> np.ndarray:
    """Return a unit noise whose correlation is ``phi**(|dx| + |dy|)``."""
    burn = 100
    white = np.random.default_rng(seed).standard_normal(
        (SHAPE[0] + burn, SHAPE[1] + burn)
    )
    gain = np.sqrt(1.0 - NOISE_CORRELATION**2)
    field = lfilter([gain], [1.0, -NOISE_CORRELATION], white, axis=1)
    field = lfilter([gain], [1.0, -NOISE_CORRELATION], field, axis=0)

    return field[burn:, burn:]


def _field(
    seed: int,
    *,
    length: float = FIELD_LENGTH,
    variance: float = FIELD_VARIANCE,
) -> np.ndarray:
    """Return a Gaussian random field of exponential correlation ``exp(-r/l)``.

    In two dimensions that correlation has the spectrum
    ``(1 + (k l)^2)^(-3/2)``.
    """
    rng = np.random.default_rng(seed)
    ky = np.fft.fftfreq(SHAPE[0])[:, None]
    kx = np.fft.fftfreq(SHAPE[1])[None, :]
    wavenumber = 2.0 * np.pi * np.hypot(kx, ky)
    amplitude = (1.0 + (wavenumber * length) ** 2) ** -0.75
    field = np.fft.ifft2(
        np.fft.fft2(rng.standard_normal(SHAPE)) * amplitude
    ).real

    return field * np.sqrt(variance) / field.std()


def _sky() -> np.ndarray:
    """Return a ring of sky around a masked galaxy of radius 100 px."""
    yy, xx = np.indices(SHAPE)

    return np.hypot(xx - 200.0, yy - 200.0) > 100.0


def _lags(kernel: np.ndarray) -> np.ndarray:
    """Return the correlation at lags 1 to 3, averaged over the two axes."""
    centre = CORRELATION_MAX_LAG

    return np.array(
        [
            0.5 * (kernel[centre, centre + lag] + kernel[centre + lag, centre])
            for lag in (1, 2, 3)
        ]
    )


@pytest.fixture(scope="module")
def estimates():
    """Estimate every sky of the six realizations once."""
    sky = _sky()
    records = []

    for seed in SEEDS:
        noise = _noise(seed)
        with_field = noise + _field(seed + 50)
        record = {
            "reference": calibrate_correlated_gaussian_noise(noise, sky),
            "noise": calibrate_correlated_gaussian_noise(
                noise,
                sky,
                estimator="field_separated",
            ),
            "field": calibrate_correlated_gaussian_noise(
                with_field,
                sky,
                estimator="field_separated",
            ),
        }

        try:
            record["autocovariance_with_field"] = (
                calibrate_correlated_gaussian_noise(with_field, sky)
            )
        except ValueError as error:
            record["autocovariance_with_field"] = error

        records.append(record)

    return records


def _ratio(record, name):
    return (
        record[name].metadata["kernel_sum"]
        / record["reference"].metadata["kernel_sum"]
    )


# ---------------------------------------------------------------------------
# The estimates
# ---------------------------------------------------------------------------


def test_without_a_field_the_estimate_is_that_of_the_autocovariance(estimates):
    """Criteria D1 and D3 of the development of the estimator."""
    ratios = [_ratio(record, "noise") for record in estimates]
    lag_errors = [
        _lags(record["noise"].correlation_kernel)
        - _lags(record["reference"].correlation_kernel)
        for record in estimates
    ]

    assert abs(np.mean(ratios) - 1.0) < 0.05
    assert all(abs(ratio - 1.0) < 0.15 for ratio in ratios)
    assert np.max(np.abs(np.mean(lag_errors, axis=0))) < 0.01


def test_a_large_scale_field_inflates_or_breaks_the_autocovariance(estimates):
    for record in estimates:
        estimate = record["autocovariance_with_field"]

        if isinstance(estimate, ValueError):
            assert "not positive definite" in str(estimate)
        else:
            assert _ratio(record, "autocovariance_with_field") > 2.0


def test_the_field_separated_estimate_recovers_the_noise_under_a_field(estimates):
    """Criteria D2 and D3: the reference is the noise alone on the same sky."""
    ratios = [_ratio(record, "field") for record in estimates]
    lag_errors = [
        _lags(record["field"].correlation_kernel)
        - _lags(record["reference"].correlation_kernel)
        for record in estimates
    ]

    assert abs(np.mean(ratios) - 1.0) < 0.10
    assert all(abs(ratio - 1.0) < 0.25 for ratio in ratios)
    assert np.max(np.abs(np.mean(lag_errors, axis=0))) < 0.01

    for record in estimates:
        metadata = record["field"].metadata
        assert metadata["field_length"] > 20.0
        assert metadata["field_amplitude"] > 0.0


def test_the_attribution_ratio_measures_the_structure_counted_as_field(
    estimates,
):
    for record in estimates:
        noise = record["noise"].metadata
        field = record["field"].metadata

        assert noise["attribution_ratio"] == pytest.approx(
            noise["high_pass_kernel_sum"] / noise["kernel_sum"]
        )
        assert abs(noise["attribution_ratio"] - 1.0) < 0.1
        assert field["attribution_ratio"] > 1.3


def test_the_kernel_is_exactly_normalized_and_symmetric(estimates):
    for record in estimates:
        for name in ("noise", "field"):
            kernel = record[name].correlation_kernel
            centre = CORRELATION_MAX_LAG

            assert kernel.shape == (2 * centre + 1, 2 * centre + 1)
            assert kernel[centre, centre] == 1.0
            assert np.array_equal(kernel, kernel[::-1, ::-1])
            assert record[name].metadata["minimum_spectrum"] > 0.0


def test_the_length_of_the_field_is_bounded_below(estimates):
    lengths = []

    for record in estimates:
        for name in ("noise", "field"):
            metadata = record[name].metadata
            lengths.append(metadata["field_length"])

            assert metadata["field_length"] >= FIELD_LENGTH_BOUNDS[0]
            assert metadata["field_length"] <= FIELD_LENGTH_BOUNDS[1]
            assert metadata["field_length_at_lower_bound"] == (
                metadata["field_length"] <= FIELD_LENGTH_BOUNDS[0] * (1 + 1e-6)
            )

    white = np.random.default_rng(3).standard_normal(SHAPE)
    metadata = calibrate_correlated_gaussian_noise(
        white,
        _sky(),
        estimator="field_separated",
    ).metadata

    assert metadata["field_length"] >= FIELD_LENGTH_BOUNDS[0]


def test_the_calibration_records_its_estimator_and_constants(estimates):
    metadata = estimates[0]["field"].metadata
    reference = estimates[0]["reference"].metadata

    assert metadata["schema_version"] == CALIBRATION_SCHEMA_VERSION
    assert metadata["estimator"] == FIELD_SEPARATED_ESTIMATOR
    assert metadata["decision"] == "AD-051"
    assert metadata["field_model"] == "v exp(-r / l) + c"
    assert metadata["field_fit_max_lag"] == 48
    assert metadata["field_length_bounds"] == list(FIELD_LENGTH_BOUNDS)
    assert metadata["field_fit_radii"][0] == CORRELATION_MAX_LAG + 1
    assert metadata["field_fit_radii"][-1] == 48
    assert metadata["noise_fraction"] == pytest.approx(
        1.0 - metadata["field_amplitude"] - metadata["field_offset"]
    )
    assert metadata["high_pass_sigma"] == 16.0
    assert metadata["max_lag"] == CORRELATION_MAX_LAG
    assert len(metadata["correlation_along_x"]) == 17
    assert reference["estimator"] == CORRELATION_ESTIMATOR
    assert reference["decision"] == "AD-047"
    assert "attribution_ratio" not in reference
    assert "field_length" not in reference


def test_the_rejection_of_the_autocovariance_names_the_declaration():
    sky = _sky()
    image = _noise(0) + _field(7, length=60.0, variance=0.2)

    with pytest.raises(ValueError, match="not positive definite") as error:
        calibrate_correlated_gaussian_noise(image, sky)

    message = str(error.value)
    assert "preparation.noise.sky_correlation: field_separated" in message
    assert (
        "preparation.masking.contaminant_growth_pixels: "
        f"{FIELD_SEPARATED_GROWTH_PIXELS}"
    ) in message


def test_a_sky_without_noise_is_rejected():
    with pytest.raises(ValueError, match="leaves no variance to the noise"):
        calibrate_correlated_gaussian_noise(
            _field(9, length=60.0, variance=1.0),
            _sky(),
            estimator="field_separated",
        )


def test_an_unknown_estimator_is_rejected():
    with pytest.raises(ValueError, match="Unknown sky-correlation estimator"):
        calibrate_correlated_gaussian_noise(
            _noise(0),
            _sky(),
            estimator="high_pass",
        )


# ---------------------------------------------------------------------------
# Persistence and declaration
# ---------------------------------------------------------------------------


def test_a_field_separated_calibration_survives_the_round_trip(
    tmp_path: Path,
    estimates,
):
    calibration = estimates[0]["field"]
    path = tmp_path / "correlated_gaussian_noise.npz"

    save_correlated_noise_calibration(path, calibration)
    loaded = load_correlated_noise_calibration(path)

    np.testing.assert_array_equal(
        loaded.correlation_kernel,
        calibration.correlation_kernel,
    )
    assert loaded.metadata == calibration.metadata


@pytest.mark.parametrize(
    "metadata, message",
    [
        (
            {"schema_version": 2, "estimator": CORRELATION_ESTIMATOR},
            "accepts only schema 3",
        ),
        (
            {"schema_version": 3, "estimator": "tapered_tiles"},
            "unknown estimator",
        ),
    ],
)
def test_a_calibration_of_another_schema_or_estimator_is_rejected(
    tmp_path: Path,
    metadata,
    message,
):
    path = tmp_path / "correlated_gaussian_noise.npz"
    np.savez_compressed(
        path,
        correlation_kernel=np.ones((1, 1)),
        metadata_json=np.asarray(json.dumps(metadata)),
    )

    with pytest.raises(ValueError, match=message):
        load_correlated_noise_calibration(path)


def _declaring(estimator: str | None) -> dict:
    config = {"preparation": {"noise": {}}}

    if estimator is not None:
        config["preparation"]["noise"]["sky_correlation"] = estimator

    return config


def test_the_declared_estimator_defaults_to_the_autocovariance():
    assert configured_sky_correlation({}) == "autocovariance"
    assert configured_sky_correlation(_declaring(None)) == "autocovariance"
    assert (
        configured_sky_correlation(_declaring("field_separated"))
        == "field_separated"
    )

    with pytest.raises(ValueError, match="sky_correlation"):
        configured_sky_correlation(_declaring("high_pass"))


def test_a_calibration_of_another_estimator_than_the_declared_one_is_rejected(
    estimates,
):
    field_separated = estimates[0]["field"]
    autocovariance = estimates[0]["reference"]

    validate_correlated_noise_calibration_estimator(
        field_separated,
        _declaring("field_separated"),
    )
    validate_correlated_noise_calibration_estimator(
        autocovariance,
        _declaring(None),
    )

    with pytest.raises(ValueError, match="declares"):
        validate_correlated_noise_calibration_estimator(
            autocovariance,
            _declaring("field_separated"),
        )

    with pytest.raises(ValueError, match="declares"):
        validate_correlated_noise_calibration_estimator(
            field_separated,
            _declaring(None),
        )


def _products(growth: int):
    """Return preparation products whose sky is the noise with a field."""
    image = 100.0 + 2.0 * (_noise(1) + _field(51))

    return SimpleNamespace(
        cutout=image,
        prepared_data=image - 100.0,
        background=100.0,
        sky_region=SimpleNamespace(mask=_sky()),
        fitting_mask=~_sky(),
        contaminant_growth_pixels=growth,
    )


def test_the_calibration_applies_the_declared_estimator(monkeypatch):
    monkeypatch.setattr(
        noise_calibration,
        "load_image_preparation_products",
        lambda config: _products(FIELD_SEPARATED_GROWTH_PIXELS),
    )
    config = _declaring("field_separated")
    config["outputs"] = {"overwrite": False}

    calibration = build_correlated_noise_calibration(config, save=False)

    assert calibration.metadata["estimator"] == FIELD_SEPARATED_ESTIMATOR
    assert calibration.metadata["generator_identity"]["estimators"] == sorted(
        [CORRELATION_ESTIMATOR, FIELD_SEPARATED_ESTIMATOR]
    )
    validate_correlated_noise_calibration_estimator(calibration, config)


def test_the_field_separated_estimator_requires_the_validated_growth(
    monkeypatch,
):
    monkeypatch.setattr(
        noise_calibration,
        "load_image_preparation_products",
        lambda config: _products(0),
    )

    with pytest.raises(ValueError, match="grown by 20 px"):
        build_correlated_noise_calibration(
            _declaring("field_separated"),
            save=False,
        )
