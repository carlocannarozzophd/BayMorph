"""Tests for generic observation-indexed representation construction."""

from __future__ import annotations

import pytest

from baymorph.fitting.representations import RepresentationFactory


class DummyRepresentation:
    """Minimal renderable representation used by unit tests."""

    def __init__(
        self,
        *,
        amplitude: float,
        scale: float,
        label: str,
    ) -> None:
        self.amplitude = amplitude
        self.scale = scale
        self.label = label

    def render(
        self,
        shape,
    ):
        del shape
        return self.amplitude * self.scale


def dummy_factory(
    *,
    amplitude,
    scale,
    label,
):
    return DummyRepresentation(
        amplitude=amplitude,
        scale=scale,
        label=label,
    )


def make_factory() -> RepresentationFactory:
    return RepresentationFactory(
        factory=dummy_factory,
        fixed_kwargs={
            "g": {
                "scale": 1.0,
                "label": "blue",
            },
            "r": {
                "scale": 2.0,
                "label": "red",
            },
        },
    )


def test_representation_factory_accepts_arbitrary_model_factory():
    factory = make_factory()

    representation = factory.build(
        "r",
        {
            "amplitude": 3.0,
        },
    )

    assert isinstance(
        representation,
        DummyRepresentation,
    )
    assert representation.amplitude == 3.0
    assert representation.scale == 2.0
    assert representation.label == "red"


def test_representation_factory_builds_all_observations():
    factory = make_factory()

    representations = factory.build_all(
        {
            "g": {
                "amplitude": 3.0,
            },
            "r": {
                "amplitude": 4.0,
            },
        }
    )

    assert tuple(representations) == (
        "g",
        "r",
    )
    assert representations["g"].scale == 1.0
    assert representations["r"].scale == 2.0


def test_representation_factory_rejects_fixed_parameter_override():
    factory = make_factory()

    with pytest.raises(
        ValueError,
        match="cannot overwrite",
    ):
        factory.build(
            "g",
            {
                "amplitude": 3.0,
                "scale": 10.0,
            },
        )


def test_representation_factory_requires_exact_observations():
    factory = make_factory()

    with pytest.raises(
        ValueError,
        match="exactly the configured observations",
    ):
        factory.build_all(
            {
                "g": {
                    "amplitude": 3.0,
                },
            }
        )


def test_factory_applies_observation_parameter_transform():
    received = {}

    def factory(
        *,
        re,
        x0,
        psf,
    ):
        received.update(
            {
                "re": re,
                "x0": x0,
                "psf": psf,
            }
        )
        return object()

    representation_factory = RepresentationFactory(
        factory=factory,
        fixed_kwargs={
            "g": {
                "psf": "g_psf",
            },
        },
        parameter_transforms={
            "g": lambda parameters: {
                "re": 2.0 * parameters["re_arcsec"],
                "x0": parameters["ra_offset"] + 10.0,
            },
        },
    )

    representation_factory.build(
        "g",
        {
            "re_arcsec": 3.0,
            "ra_offset": 1.5,
        },
    )

    assert received == {
        "re": 6.0,
        "x0": 11.5,
        "psf": "g_psf",
    }


def test_factory_leaves_untransformed_observations_unchanged():
    received = {}

    def factory(
        *,
        re,
        psf,
    ):
        received.update(
            {
                "re": re,
                "psf": psf,
            }
        )
        return object()

    representation_factory = RepresentationFactory(
        factory=factory,
        fixed_kwargs={
            "g": {
                "psf": "g_psf",
            },
        },
    )

    representation_factory.build(
        "g",
        {
            "re": 4.0,
        },
    )

    assert received == {
        "re": 4.0,
        "psf": "g_psf",
    }


def test_factory_rejects_transform_for_unknown_observation():
    with pytest.raises(
        ValueError,
        match="unknown observations",
    ):
        RepresentationFactory(
            factory=lambda **kwargs: kwargs,
            fixed_kwargs={
                "g": {},
            },
            parameter_transforms={
                "r": lambda parameters: parameters,
            },
        )


def test_factory_rejects_non_mapping_transform_result():
    representation_factory = RepresentationFactory(
        factory=lambda **kwargs: kwargs,
        fixed_kwargs={
            "g": {},
        },
        parameter_transforms={
            "g": lambda parameters: 1.0,
        },
    )

    with pytest.raises(
        TypeError,
        match="must return mappings",
    ):
        representation_factory.build(
            "g",
            {
                "re": 4.0,
            },
        )
