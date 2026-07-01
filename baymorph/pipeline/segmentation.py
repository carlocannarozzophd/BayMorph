"""Segmentation utilities."""

import numpy as np


def select_segment_near_position(
    segmentation,
    x: float,
    y: float,
):
    """Select the segmentation label closest to a target position.

    The function first checks whether the requested position falls inside a
    detected segment. If not, it selects the detected segment whose centroid
    is closest to the requested position.

    Parameters
    ----------
    segmentation : photutils.segmentation.SegmentationImage
        Segmentation image produced by Photutils.
    x : float
        Approximate target x-coordinate, in pixels.
    y : float
        Approximate target y-coordinate, in pixels.

    Returns
    -------
    int
        Selected segmentation label.

    Raises
    ------
    ValueError
        If the segmentation image is ``None`` or contains no detected labels.
    """
    if segmentation is None:
        raise ValueError(
            "Segmentation image is None."
        )

    labels = np.unique(
        segmentation.data
    )

    labels = labels[
        labels > 0
    ]

    if len(labels) == 0:
        raise ValueError(
            "No segmentation labels found."
        )

    x = int(
        round(x)
    )

    y = int(
        round(y)
    )

    if (
        0 <= y < segmentation.data.shape[0]
        and 0 <= x < segmentation.data.shape[1]
    ):
        label = segmentation.data[
            y,
            x,
        ]

        if label > 0:
            return int(label)

    best_label = None
    best_distance = np.inf

    for label in labels:
        pixels = np.where(
            segmentation.data == label
        )

        y_c = np.mean(
            pixels[0]
        )

        x_c = np.mean(
            pixels[1]
        )

        distance = np.hypot(
            x_c - x,
            y_c - y,
        )

        if distance < best_distance:
            best_distance = distance
            best_label = label

    return int(
        best_label
    )