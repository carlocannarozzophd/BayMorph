import pytest

from baymorph.validation import SimulationGrid


def test_simulation_grid_builds_deterministic_cartesian_product():
    grid = SimulationGrid({"snr": [10, 20], "n": [1.0, 4.0]})

    assert len(grid) == 4
    assert [case.parameters for case in grid] == [
        {"snr": 10, "n": 1.0},
        {"snr": 10, "n": 4.0},
        {"snr": 20, "n": 1.0},
        {"snr": 20, "n": 4.0},
    ]
    assert grid.case(2).identifier == "case-00002"


def test_simulation_grid_rejects_empty_axes():
    with pytest.raises(ValueError):
        SimulationGrid({"snr": []})


def test_simulation_grid_rejects_out_of_range_case():
    grid = SimulationGrid({"snr": [10]})

    with pytest.raises(IndexError):
        grid.case(1)
