import numpy as np
from photutils.segmentation import SegmentationImage

from baymorph.pipeline.segmentation import (
    select_segment_near_position,
)


class MockSegmentation:

    def __init__(
        self,
        data,
    ):
        self.data = data


def test_select_segment_near_position():

    seg = MockSegmentation(
        np.array(
            [
                [0, 0, 0, 0],
                [0, 1, 1, 0],
                [0, 1, 1, 0],
                [0, 0, 2, 2],
            ]
        )
    )

    label = (
        select_segment_near_position(
            seg,
            x=1,
            y=1,
        )
    )

    assert label == 1


def test_select_segment_uses_nearest_footprint_not_centroid():
    """An extended target footprint should beat a closer compact centroid."""
    segmentation = SegmentationImage(
        np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0, 0, 0, 0, 2, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 2, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=int,
        )
    )

    label = select_segment_near_position(
        segmentation,
        x=6.2,
        y=4.7,
    )

    assert label == 1