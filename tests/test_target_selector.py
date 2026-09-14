"""Tests for interactive target-selection state."""

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits

from baymorph import ObservationAdvisor
from baymorph.plotting.target_selector import TargetSelector


def test_target_selector_requires_explicit_confirmation() -> None:
    """A candidate must not become the scientific target implicitly."""
    confirmed = []

    selector = TargetSelector(
        np.ones(
            (
                5,
                7,
            )
        ),
        on_confirm=confirmed.append,
    )

    assert selector.candidate == (
        3.0,
        2.0,
    )
    assert selector.value is None
    assert not selector.is_confirmed

    selector.set_candidate(
        1.25,
        3.5,
    )

    assert selector.value is None

    position = selector.confirm()

    assert position == (
        1.25,
        3.5,
    )
    assert selector.value == position
    assert selector.is_confirmed
    assert confirmed == [position]


def test_changing_confirmed_candidate_invalidates_selection() -> None:
    """Moving a confirmed marker must require a new confirmation."""
    invalidations = []

    selector = TargetSelector(
        np.ones(
            (
                6,
                8,
            )
        ),
        initial_position=(
            2.0,
            3.0,
        ),
        on_invalidate=lambda: invalidations.append(True),
    )

    selector.confirm()
    selector.set_candidate(
        2.5,
        3.5,
    )

    assert selector.value is None
    assert not selector.is_confirmed
    assert invalidations == [True]


def test_target_selector_reset_restores_initial_candidate() -> None:
    """Reset must restore the initial point without confirming it."""
    selector = TargetSelector(
        np.ones(
            (
                10,
                12,
            )
        ),
        initial_position=(
            4.5,
            6.5,
        ),
    )

    selector.set_candidate(
        1.0,
        2.0,
    )
    selector.confirm()
    selector.reset()

    assert selector.candidate == (
        4.5,
        6.5,
    )
    assert selector.value is None


@pytest.mark.parametrize(
    "position",
    [
        (
            -0.1,
            2.0,
        ),
        (
            7.0,
            2.0,
        ),
        (
            2.0,
            -0.1,
        ),
        (
            2.0,
            5.0,
        ),
        (
            np.nan,
            2.0,
        ),
    ],
)
def test_target_selector_rejects_invalid_coordinates(
    position: tuple[float, float],
) -> None:
    """Scientific target coordinates must be finite and inside the image."""
    selector = TargetSelector(
        np.ones(
            (
                5,
                7,
            )
        )
    )

    with pytest.raises(ValueError):
        selector.set_candidate(*position)


def test_target_selector_requires_a_finite_2d_image() -> None:
    """The selector must reject unsupported or unusable image arrays."""
    with pytest.raises(
        ValueError,
        match="two-dimensional",
    ):
        TargetSelector(
            np.ones(
                (
                    2,
                    3,
                    4,
                )
            )
        )

    with pytest.raises(
        ValueError,
        match="finite pixel",
    ):
        TargetSelector(
            np.full(
                (
                    4,
                    4,
                ),
                np.nan,
            )
        )


def test_advisor_target_selection_requires_confirmation(
    tmp_path,
) -> None:
    """The Advisor must use only the explicitly confirmed widget position."""
    fits_path = tmp_path / "observation.fits"

    fits.writeto(
        fits_path,
        np.arange(35.0).reshape(
            5,
            7,
        ),
    )

    advisor = ObservationAdvisor(fits_path)
    selector = advisor.select_target()

    assert selector.candidate == (
        3.0,
        2.0,
    )
    assert selector.value is None
    assert advisor.target_position is None

    selector.set_candidate(
        1.25,
        3.5,
    )

    assert advisor.target_position is None

    selector.confirm()

    assert advisor.target_position == (
        1.25,
        3.5,
    )

    selector.set_candidate(
        2.0,
        3.0,
    )

    assert selector.value is None
    assert advisor.target_position is None


def test_advisor_target_selector_preserves_original_pixel_frame(
    tmp_path,
) -> None:
    """Preview sampling must not change the selected FITS coordinates."""
    fits_path = tmp_path / "large_observation.fits"

    fits.writeto(
        fits_path,
        np.ones(
            (
                120,
                200,
            )
        ),
    )

    advisor = ObservationAdvisor(fits_path)
    selector = advisor.select_target(max_preview_size=20)

    selector.set_candidate(
        173.25,
        91.75,
    )
    selector.confirm()

    assert advisor.target_position == (
        173.25,
        91.75,
    )


def test_advisor_rejects_target_outside_fits_image(
    tmp_path,
) -> None:
    """Manual Advisor coordinates must obey the same image bounds."""
    fits_path = tmp_path / "observation.fits"

    fits.writeto(
        fits_path,
        np.ones(
            (
                10,
                12,
            )
        ),
    )

    advisor = ObservationAdvisor(fits_path)

    with pytest.raises(
        ValueError,
        match="inside the FITS image",
    ):
        advisor.set_target_position(
            12.0,
            5.0,
        )
