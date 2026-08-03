import pytest

from baymorph.rendering import IntegrationCell


def test_integration_cell_geometry():
    cell = IntegrationCell(
        xmin=2.0,
        xmax=6.0,
        ymin=-1.0,
        ymax=2.0,
    )

    assert cell.width == pytest.approx(4.0)
    assert cell.height == pytest.approx(3.0)
    assert cell.area == pytest.approx(12.0)
    assert cell.center == pytest.approx((4.0, 0.5))
    assert cell.level == 0


def test_integration_cell_subdivision():
    cell = IntegrationCell(
        xmin=0.0,
        xmax=2.0,
        ymin=0.0,
        ymax=4.0,
        level=3,
    )

    children = cell.subdivide()

    assert len(children) == 4
    assert all(child.level == 4 for child in children)
    assert all(child.width == pytest.approx(1.0) for child in children)
    assert all(child.height == pytest.approx(2.0) for child in children)

    assert children == (
        IntegrationCell(0.0, 1.0, 0.0, 2.0, level=4),
        IntegrationCell(1.0, 2.0, 0.0, 2.0, level=4),
        IntegrationCell(0.0, 1.0, 2.0, 4.0, level=4),
        IntegrationCell(1.0, 2.0, 2.0, 4.0, level=4),
    )


def test_integration_cell_subdivision_preserves_area():
    cell = IntegrationCell(
        xmin=-2.0,
        xmax=3.0,
        ymin=4.0,
        ymax=6.0,
    )

    children = cell.subdivide()

    assert sum(child.area for child in children) == pytest.approx(cell.area)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {
                "xmin": 1.0,
                "xmax": 1.0,
                "ymin": 0.0,
                "ymax": 1.0,
            },
            "xmax must be greater than xmin",
        ),
        (
            {
                "xmin": 2.0,
                "xmax": 1.0,
                "ymin": 0.0,
                "ymax": 1.0,
            },
            "xmax must be greater than xmin",
        ),
        (
            {
                "xmin": 0.0,
                "xmax": 1.0,
                "ymin": 1.0,
                "ymax": 1.0,
            },
            "ymax must be greater than ymin",
        ),
        (
            {
                "xmin": 0.0,
                "xmax": 1.0,
                "ymin": 2.0,
                "ymax": 1.0,
            },
            "ymax must be greater than ymin",
        ),
        (
            {
                "xmin": 0.0,
                "xmax": 1.0,
                "ymin": 0.0,
                "ymax": 1.0,
                "level": -1,
            },
            "level must be non-negative",
        ),
    ],
)
def test_integration_cell_rejects_invalid_geometry(kwargs, message):
    with pytest.raises(ValueError, match=message):
        IntegrationCell(**kwargs)
