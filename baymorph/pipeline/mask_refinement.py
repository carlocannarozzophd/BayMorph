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
    """
    Refine a boolean mask using reproducible masking actions.

    Parameters
    ----------
    image : ndarray
        Input image.

    mask : ndarray of bool
        Initial mask.

    actions : list of dict
        List of masking actions.

    Returns
    -------
    refined_mask : ndarray of bool
        Refined boolean mask.
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