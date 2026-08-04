import pytest

from baymorph.fitting.problem import InferenceProblem


def test_inference_problem_is_abstract():
    with pytest.raises(TypeError):
        InferenceProblem()
