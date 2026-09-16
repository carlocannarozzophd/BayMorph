"""Utilities for reproducible mask refinement."""

from collections.abc import Mapping

import numpy as np

from baymorph.core import (
    build_elliptical_radius_map,
    connected_source_mask,
    star_halo_mask,
)


def _ellipse_mask(
    shape: tuple[int, int],
    action: Mapping,
) -> np.ndarray:
    """Build a boolean elliptical region from one refinement action."""
    x = float(action["x"])
    y = float(action["y"])
    semi_major = float(action["semi_major"])
    axis_ratio = float(action.get("axis_ratio", 1.0))
    theta = float(action.get("theta", 0.0))

    values = np.asarray(
        [x, y, semi_major, axis_ratio, theta],
        dtype=float,
    )

    if not np.all(np.isfinite(values)):
        raise ValueError(
            "Ellipse parameters must be finite."
        )

    if semi_major <= 0.0:
        raise ValueError(
            "Ellipse semi_major must be positive."
        )

    if not 0.0 < axis_ratio <= 1.0:
        raise ValueError(
            "Ellipse axis_ratio must satisfy 0 < q <= 1."
        )

    radius_map = build_elliptical_radius_map(
        shape=shape,
        x0=x,
        y0=y,
        q=axis_ratio,
        theta=theta,
    )

    return radius_map <= semi_major


def refine_mask(
    image,
    mask,
    actions,
):
    """Apply reproducible masking and unmasking actions sequentially.

    Supported action types are ``source``, ``star``, and ``ellipse``.
    Every action accepts ``operation: mask`` or ``operation: unmask``.
    The default operation remains ``mask`` for backward compatibility.
    """
    image_array = np.asarray(
        image,
        dtype=float,
    )
    refined_mask = np.array(
        mask,
        dtype=bool,
        copy=True,
    )

    if image_array.ndim != 2:
        raise ValueError(
            "Mask refinement requires a two-dimensional image."
        )

    if refined_mask.shape != image_array.shape:
        raise ValueError(
            "Image and initial mask must have identical shapes."
        )

    for index, action in enumerate(actions):
        if not isinstance(action, Mapping):
            raise TypeError(
                f"Mask refinement action {index} must be a mapping."
            )

        action_type = str(
            action["type"]
        ).strip().lower()

        operation = str(
            action.get("operation", "mask")
        ).strip().lower()

        if operation not in {"mask", "unmask"}:
            raise ValueError(
                "Mask refinement operation must be "
                "'mask' or 'unmask'."
            )

        if action_type == "source":
            action_mask = connected_source_mask(
                image=image_array,
                x=action["x"],
                y=action["y"],
                threshold_sigma=action.get(
                    "threshold_sigma",
                    2.0,
                ),
                n_pixels=action.get(
                    "n_pixels",
                    5,
                ),
                dilation_iterations=action.get(
                    "dilation_iterations",
                    0,
                ),
                box_size=action.get(
                    "box_size",
                    101,
                ),
            )

        elif action_type == "star":
            action_mask = star_halo_mask(
                image=image_array,
                x=action["x"],
                y=action["y"],
                radius=action.get(
                    "radius",
                    50,
                ),
                threshold_sigma=action.get(
                    "threshold_sigma",
                    2.0,
                ),
                n_pixels=action.get(
                    "n_pixels",
                    5,
                ),
                dilation_iterations=action.get(
                    "dilation_iterations",
                    10,
                ),
            )

        elif action_type == "ellipse":
            action_mask = _ellipse_mask(
                image_array.shape,
                action,
            )

        else:
            raise ValueError(
                "Unknown mask refinement action type: "
                f"{action_type}"
            )

        if operation == "mask":
            refined_mask |= action_mask
        else:
            refined_mask &= ~action_mask

    return refined_mask
