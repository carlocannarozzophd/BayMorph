"""Utilities for reproducible mask refinement."""

import numpy as np

from baymorph.core import (
    connected_source_mask,
    star_halo_mask,
)


def refine_mask(
    image,
    mask,
    actions,
):
    """Refine a boolean mask using a sequence of masking actions.

    Each action is represented as a dictionary describing a masking operation.
    Actions are applied sequentially, allowing interactive mask editing to be
    reproduced deterministically.

    Parameters
    ----------
    image : ndarray
        Input image.
    mask : ndarray of bool
        Initial boolean mask.
    actions : list of dict
        Sequence of masking actions. Each action must contain a ``"type"``
        entry identifying the masking operation together with the parameters
        required by that operation.

    Returns
    -------
    ndarray of bool
        Refined boolean mask obtained after applying all masking actions.

    Raises
    ------
    ValueError
        If an action specifies an unsupported masking type.
    """
    refined_mask = np.array(
        mask,
        dtype=bool,
        copy=True,
    )

    for action in actions:
        action_type = action["type"]

        if action_type == "source":
            extra_mask = connected_source_mask(
                image=image,
                x=action["x"],
                y=action["y"],
                threshold_sigma=action.get("threshold_sigma", 2.0),
                n_pixels=action.get("n_pixels", 5),
                dilation_iterations=action.get(
                    "dilation_iterations",
                    0,
                ),
                box_size=action.get("box_size", 101),
            )

        elif action_type == "star":
            extra_mask = star_halo_mask(
                image=image,
                x=action["x"],
                y=action["y"],
                radius=action.get("radius", 50),
                threshold_sigma=action.get("threshold_sigma", 2.0),
                n_pixels=action.get("n_pixels", 5),
                dilation_iterations=action.get(
                    "dilation_iterations",
                    10,
                ),
            )

        else:
            raise ValueError(
                f"Unknown mask refinement action type: {action_type}"
            )

        refined_mask |= extra_mask

    return refined_mask