from __future__ import annotations

import numpy as np
import pytest

from baymorph.fitting.geometry import GeometryModel, LogTransform
from baymorph.fitting.geometry import ParameterGeometry
from baymorph.fitting.geometry.nested.live_points import LivePoints
from baymorph.fitting.geometry.nested.sampler import GeometryNestedSampler
from baymorph.fitting.parameters import Parameter, ParameterSet


def make_geometry() -> GeometryModel:
    return GeometryModel(
        [
            ParameterGeometry(
                "scale",
                transform=LogTransform(),
                block="shape",
            ),
            ParameterGeometry(
                "angle",
                block="orientation",
                period=float(np.pi),
            ),
        ]
    )


def make_parameters() -> ParameterSet:
    return ParameterSet(
        [
            Parameter("scale", 1.0, 3.0),
            Parameter("angle", 0.0, float(np.pi)),
        ]
    )


def test_geometry_package_exposes_foundation_types():
    geometry = make_geometry()

    assert geometry.names == ["scale", "angle"]
    assert geometry.ndim == 2


def test_geometry_transform_round_trip():
    geometry = make_geometry()
    physical = np.array([2.0, 0.5])

    proposal = geometry.to_proposal_space(physical)
    recovered = geometry.from_proposal_space(proposal)

    assert np.allclose(recovered, physical)


def test_geometry_rejects_invalid_vector_shape():
    geometry = make_geometry()

    with pytest.raises(ValueError, match="Expected 2 physical values"):
        geometry.to_proposal_space([1.0])


def test_log_transform_rejects_non_positive_values():
    transform = LogTransform()

    with pytest.raises(ValueError, match="positive finite"):
        transform.forward(0.0)

    with pytest.raises(ValueError, match="positive finite"):
        transform.forward(-1.0)


def test_geometry_rejects_duplicate_names():
    with pytest.raises(ValueError, match="must be unique"):
        GeometryModel(
            [
                ParameterGeometry("value"),
                ParameterGeometry("value"),
            ]
        )


def test_geometry_derives_blocks_and_periodic_indices():
    geometry = make_geometry()

    assert geometry.blocks == {
        "shape": ("scale",),
        "orientation": ("angle",),
    }
    assert geometry.block_indices("orientation") == [1]
    assert geometry.periodic == [1]
    assert geometry.geometry("angle").period == pytest.approx(np.pi)
    assert [block.name for block in geometry.proposal_blocks] == [
        "shape",
        "orientation",
    ]


def test_geometry_validates_parameter_ordering():
    geometry = make_geometry()

    with pytest.raises(ValueError, match="parameter ordering must match"):
        geometry.validate_parameter_names(["angle", "scale"])


def test_live_points_validate_shape_and_nan_logl():
    live_points = LivePoints(make_geometry())

    with pytest.raises(ValueError, match="Expected live-point shape"):
        live_points.append([1.0], 0.0)

    with pytest.raises(ValueError, match="must not be NaN"):
        live_points.append([1.0, 0.0], np.nan)


def test_live_points_empty_access_is_explicit():
    live_points = LivePoints(make_geometry())

    assert live_points.values.shape == (0, 2)
    with pytest.raises(ValueError, match="LivePoints is empty"):
        _ = live_points.worst


def test_live_points_covariance_uses_proposal_space_by_default():
    geometry = make_geometry()
    live_points = LivePoints(geometry)
    live_points.append([1.0, 0.0], -2.0)
    live_points.append([np.e, 1.0], -1.0)

    covariance = live_points.covariance()
    expected = np.cov(np.array([[0.0, 0.0], [1.0, 1.0]]), rowvar=False)

    assert np.allclose(covariance, expected)


def test_sampler_initialization_is_deterministic_and_counted():
    def likelihood(values):
        return -float(np.sum(values**2))

    first = GeometryNestedSampler(
        geometry=make_geometry(),
        parameter_set=make_parameters(),
        likelihood=likelihood,
        nlive=5,
        seed=17,
    )
    second = GeometryNestedSampler(
        geometry=make_geometry(),
        parameter_set=make_parameters(),
        likelihood=likelihood,
        nlive=5,
        seed=17,
    )

    first.initialize()
    second.initialize()

    assert first.state.initialized is True
    assert first.state.ncall == 5
    assert len(first.live_points) == 5
    assert np.allclose(first.live_points.values, second.live_points.values)
    assert np.all(np.diff(first.live_points.logl) >= 0.0)


def test_sampler_rejects_geometry_parameter_mismatch():
    parameters = ParameterSet(
        [
            Parameter("angle", 0.0, 1.0),
            Parameter("scale", 1.0, 2.0),
        ]
    )

    with pytest.raises(ValueError, match="parameter ordering must match"):
        GeometryNestedSampler(
            geometry=make_geometry(),
            parameter_set=parameters,
            likelihood=lambda values: 0.0,
        )


def test_sampler_cannot_be_initialized_twice():
    sampler = GeometryNestedSampler(
        geometry=make_geometry(),
        parameter_set=make_parameters(),
        likelihood=lambda values: 0.0,
        nlive=2,
        seed=3,
    )
    sampler.initialize()

    with pytest.raises(RuntimeError, match="already initialized"):
        sampler.initialize()
