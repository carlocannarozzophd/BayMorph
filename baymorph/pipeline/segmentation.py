"""Segmentation utilities."""

import numpy as np


def select_segment_near_position(segmentation, x: float, y: float):
    """Select the segmentation label nearest to a target position.

    The requested position is first rounded to the nearest pixel. If that
    pixel belongs to a detected segment, its label is returned directly.

    Otherwise, the selected label is the one whose segmentation footprint
    contains the pixel closest to the requested position. This is more robust
    for extended or asymmetric sources than comparing segment centroids,
    because a compact foreground source can have a closer centroid even when
    the target galaxy footprint lies closer to the supplied seed.

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
        If the segmentation image is ``None``, contains no detected labels,
        or the supplied coordinates are non-finite.
    """
    if segmentation is None:
        raise ValueError("Segmentation image is None.")

    data = np.asarray(segmentation.data)

    labels = np.unique(data)
    labels = labels[labels > 0]

    if labels.size == 0:
        raise ValueError("No segmentation labels found.")

    target_x = float(x)
    target_y = float(y)

    if not np.isfinite(target_x) or not np.isfinite(target_y):
        raise ValueError("Target coordinates must be finite.")

    pixel_x = int(round(target_x))
    pixel_y = int(round(target_y))

    if 0 <= pixel_y < data.shape[0] and 0 <= pixel_x < data.shape[1]:
        label_at_position = int(data[pixel_y, pixel_x])

        if label_at_position > 0:
            return label_at_position

    best_label: int | None = None
    best_squared_distance = np.inf

    for label in labels:
        segment_y, segment_x = np.nonzero(data == label)

        squared_distances = (segment_x - target_x) ** 2 + (segment_y - target_y) ** 2

        segment_squared_distance = float(np.min(squared_distances))

        if segment_squared_distance < best_squared_distance:
            best_squared_distance = segment_squared_distance
            best_label = int(label)

    if best_label is None:
        raise ValueError("No valid segmentation label could be selected.")

    return best_label
