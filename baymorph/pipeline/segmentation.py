"""Segmentation utilities."""

import numpy as np


def select_segment_near_position(
    segmentation,
    x: float,
    y: float,
):
    """
    Select the segmentation label nearest to a position.

    Parameters
    ----------
    segmentation
        Photutils SegmentationImage.

    x, y
        Approximate target coordinates.

    Returns
    -------
    int
        Selected segmentation label.
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
        and
        0 <= x < segmentation.data.shape[1]
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