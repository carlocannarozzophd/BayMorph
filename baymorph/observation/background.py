"""Deterministic additive background models for observed images."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def _validate_shape(
    shape: tuple[int, int],
) -> tuple[int, int]:
    """Validate and normalize a two-dimensional image shape."""
    if (
        len(shape) != 2
        or any(
            not isinstance(value, (int, np.integer))
            or value <= 0
            for value in shape
        )
    ):
        raise ValueError(
            "Background render shape must contain two positive integers."
        )

    return int(shape[0]), int(shape[1])


def _finite_float(
    value: float,
    *,
    name: str,
) -> float:
    """Return one finite floating-point background parameter."""
    value = float(value)

    if not np.isfinite(value):
        raise ValueError(
            f"{name} must be finite."
        )

    return value


@dataclass(frozen=True)
class ConstantBackground:
    """Constant residual additive background.

    Parameters
    ----------
    level : float
        Residual background level after preparation-stage reference
        subtraction.
    """

    level: float

    def __post_init__(self) -> None:
        """Validate the residual background level."""
        object.__setattr__(
            self,
            "level",
            _finite_float(
                self.level,
                name="background level",
            ),
        )

    def render(
        self,
        shape: tuple[int, int],
    ) -> FloatArray:
        """Render the residual background on one detector grid."""
        ny, nx = _validate_shape(
            shape
        )

        return np.full(
            (ny, nx),
            self.level,
            dtype=float,
        )

    def apply(
        self,
        image: ArrayLike,
    ) -> FloatArray:
        """Add the residual background to an expected source image."""
        image_array = np.asarray(
            image,
            dtype=float,
        )

        if image_array.ndim != 2:
            raise ValueError(
                "Background effects require a two-dimensional image."
            )

        return (
            image_array
            + self.render(
                image_array.shape
            )
        )


@dataclass(frozen=True)
class TiltedPlaneBackground:
    """Residual background represented by a detector-frame plane.

    The model is

    ``B(x, y) = level + gradient_x * (x - x_ref)
                      + gradient_y * (y - y_ref)``,

    where the reference position is the geometric centre of the rendered
    image. Gradients are therefore expressed in image-intensity units per
    detector pixel.

    Parameters
    ----------
    level : float
        Residual background level at the geometric image centre.
    gradient_x : float
        Residual background gradient along increasing detector x.
    gradient_y : float
        Residual background gradient along increasing detector y.
    """

    level: float
    gradient_x: float
    gradient_y: float

    def __post_init__(self) -> None:
        """Validate the plane coefficients."""
        for attribute, label in (
            ("level", "background level"),
            ("gradient_x", "background gradient_x"),
            ("gradient_y", "background gradient_y"),
        ):
            object.__setattr__(
                self,
                attribute,
                _finite_float(
                    getattr(
                        self,
                        attribute,
                    ),
                    name=label,
                ),
            )

    def render(
        self,
        shape: tuple[int, int],
    ) -> FloatArray:
        """Render the residual tilted plane on one detector grid."""
        ny, nx = _validate_shape(
            shape
        )

        y, x = np.indices(
            (ny, nx),
            dtype=float,
        )

        x_ref = 0.5 * (nx - 1)
        y_ref = 0.5 * (ny - 1)

        return np.asarray(
            self.level
            + self.gradient_x
            * (
                x - x_ref
            )
            + self.gradient_y
            * (
                y - y_ref
            ),
            dtype=float,
        )

    def apply(
        self,
        image: ArrayLike,
    ) -> FloatArray:
        """Add the residual tilted plane to an expected source image."""
        image_array = np.asarray(
            image,
            dtype=float,
        )

        if image_array.ndim != 2:
            raise ValueError(
                "Background effects require a two-dimensional image."
            )

        return (
            image_array
            + self.render(
                image_array.shape
            )
        )


class BackgroundAppliedModel:
    """Apply one additive observation-level background to a source model.

    Parameters
    ----------
    model : object
        Observed source model exposing ``render(shape)`` and optionally
        ``render_components(shape)``.
    background : ConstantBackground or TiltedPlaneBackground
        Residual additive background model.

    Notes
    -----
    The background is an observation effect and is intentionally excluded
    from ``render_components()``. It is not part of the intrinsic galaxy
    morphology.
    """

    def __init__(
        self,
        *,
        model: Any,
        background: ConstantBackground | TiltedPlaneBackground,
    ) -> None:
        if not hasattr(
            model,
            "render",
        ):
            raise TypeError(
                "BackgroundAppliedModel requires a model exposing render()."
            )

        self.model = model
        self.background = background

    def render(
        self,
        shape: tuple[int, int],
    ) -> FloatArray:
        """Render the observed source plus residual background."""
        return self.background.apply(
            self.model.render(
                shape
            )
        )

    def render_prediction(
        self,
        shape: tuple[int, int],
    ) -> tuple[FloatArray, FloatArray]:
        """Render the source plus residual background, and the source alone.

        The source part is what the source Poisson variance of AD-047
        depends on; the residual background is an observation effect that
        does not enter it.
        """
        source = np.asarray(
            self.model.render(
                shape
            ),
            dtype=float,
        )

        return (
            self.background.apply(
                source
            ),
            source,
        )

    def render_components(
        self,
        shape: tuple[int, int],
    ):
        """Render source components without the additive background."""
        if not hasattr(
            self.model,
            "render_components",
        ):
            raise AttributeError(
                "Wrapped model does not expose render_components()."
            )

        return self.model.render_components(
            shape
        )

    def render_background(
        self,
        shape: tuple[int, int],
    ) -> FloatArray:
        """Render only the additive residual background."""
        return self.background.render(
            shape
        )


BACKGROUND_PARAMETER_NAMES = {
    "constant": (
        "background_level",
    ),
    "plane": (
        "background_level",
        "background_gradient_x",
        "background_gradient_y",
    ),
}


def background_parameter_names(
    model: str,
) -> tuple[str, ...]:
    """Return the ordered sampled parameters of one background family.

    Parameters
    ----------
    model : {"constant", "plane"}
        Background-model family.

    Returns
    -------
    tuple of str
        Parameter names in the deterministic order used by inference.
    """
    key = str(
        model
    ).strip().lower()

    try:
        return BACKGROUND_PARAMETER_NAMES[
            key
        ]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported background model {model!r}. "
            "Available models: 'constant', 'plane'."
        ) from exc


def make_background_effect(
    model: str,
    *,
    level: float,
    gradient_x: float = 0.0,
    gradient_y: float = 0.0,
) -> ConstantBackground | TiltedPlaneBackground:
    """Construct one residual background observation effect.

    Parameters
    ----------
    model : {"constant", "plane"}
        Background-model family.
    level : float
        Residual background level at the image centre.
    gradient_x, gradient_y : float, default=0.0
        Detector-frame gradients in image-intensity units per pixel.
        These parameters are used only by the ``plane`` model.

    Returns
    -------
    ConstantBackground or TiltedPlaneBackground
        Configured residual background effect.
    """
    model = str(
        model
    ).strip().lower()

    if model == "constant":
        if (
            float(gradient_x) != 0.0
            or float(gradient_y) != 0.0
        ):
            raise ValueError(
                "Constant background does not accept non-zero gradients."
            )

        return ConstantBackground(
            level=level,
        )

    if model == "plane":
        return TiltedPlaneBackground(
            level=level,
            gradient_x=gradient_x,
            gradient_y=gradient_y,
        )

    raise ValueError(
        f"Unsupported background model {model!r}. "
        "Available models: 'constant', 'plane'."
    )
