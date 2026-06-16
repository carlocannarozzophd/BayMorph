import numpy as np

from baymorph import (
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