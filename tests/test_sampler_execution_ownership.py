"""Tests for ownership of backend-neutral sampler execution options."""

from __future__ import annotations

import numpy as np
import pytest

from baymorph.fitting.dynesty import DynestyFitter
from baymorph.fitting.execution import SamplerExecution
from baymorph.fitting.nautilus import NautilusFitter
from baymorph.fitting.pocomc import PocoMCFitter
from baymorph.fitting.problem import InferenceProblem


class ScalarProblem(InferenceProblem):
    """Minimal inference problem for execution-ownership tests."""

    @property
    def ndim(self) -> int:
        return 1

    @property
    def parameter_names(self) -> list[str]:
        return ["x"]

    def prior_transform(
        self,
        cube: np.ndarray,
    ) -> np.ndarray:
        return np.asarray(
            cube,
            dtype=float,
        )

    def log_likelihood(
        self,
        theta: np.ndarray,
    ) -> float:
        value = float(
            np.asarray(
                theta,
                dtype=float,
            )[0]
        )

        return -0.5 * value**2


EXECUTION = SamplerExecution(
    requested_processes=2,
    processes=2,
    progress=False,
)


@pytest.mark.parametrize(
    "option",
    [
        "pool",
        "queue_size",
    ],
)
def test_dynesty_execution_owns_constructor_parallelism(
    option: str,
) -> None:
    """Dynesty native config cannot override BayMorph parallel execution."""
    fitter = DynestyFitter(
        problem=ScalarProblem(),
    )

    with pytest.raises(
        ValueError,
        match="owned by BayMorph",
    ):
        fitter.run(
            init_kwargs={
                option: object(),
            },
            execution=EXECUTION,
        )


def test_dynesty_execution_owns_progress() -> None:
    """Dynesty native run config cannot override BayMorph progress."""
    fitter = DynestyFitter(
        problem=ScalarProblem(),
    )

    with pytest.raises(
        ValueError,
        match="owned by",
    ):
        fitter.run(
            run_kwargs={
                "print_progress": False,
            },
            execution=EXECUTION,
        )


def test_pocomc_execution_owns_pool() -> None:
    """pocoMC native config cannot override BayMorph parallel execution."""
    fitter = PocoMCFitter(
        problem=ScalarProblem(),
    )

    with pytest.raises(
        ValueError,
        match="owned by",
    ):
        fitter.run(
            init_kwargs={
                "pool": object(),
            },
            execution=EXECUTION,
        )


def test_pocomc_execution_owns_progress() -> None:
    """pocoMC native run config cannot override BayMorph progress."""
    fitter = PocoMCFitter(
        problem=ScalarProblem(),
    )

    with pytest.raises(
        ValueError,
        match="owned by",
    ):
        fitter.run(
            run_kwargs={
                "progress": False,
            },
            execution=EXECUTION,
        )


def test_nautilus_execution_owns_pool() -> None:
    """Nautilus native config cannot override BayMorph parallel execution."""
    fitter = NautilusFitter(
        problem=ScalarProblem(),
    )

    with pytest.raises(
        ValueError,
        match="owned by",
    ):
        fitter.run(
            init_kwargs={
                "pool": object(),
            },
            execution=EXECUTION,
        )


def test_nautilus_execution_owns_progress() -> None:
    """Nautilus native run config cannot override BayMorph progress."""
    fitter = NautilusFitter(
        problem=ScalarProblem(),
    )

    with pytest.raises(
        ValueError,
        match="owned by",
    ):
        fitter.run(
            run_kwargs={
                "verbose": False,
            },
            execution=EXECUTION,
        )
