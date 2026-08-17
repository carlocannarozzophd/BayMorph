"""Tests for backend-neutral sampler execution configuration."""

from __future__ import annotations

import pytest

from baymorph.fitting.execution import (
    resolve_process_count,
    resolve_sampler_execution,
)


def test_auto_processes_use_ten_on_twelve_cpu_machine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Automatic local execution leaves two of twelve CPUs unused."""
    monkeypatch.setattr(
        "baymorph.fitting.execution.os.cpu_count",
        lambda: 12,
    )

    assert resolve_process_count(
        "auto"
    ) == 10


def test_auto_processes_leave_two_cpus_when_possible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Automatic execution reserves two CPUs below the ten-worker cap."""
    monkeypatch.setattr(
        "baymorph.fitting.execution.os.cpu_count",
        lambda: 8,
    )

    assert resolve_process_count(
        "auto"
    ) == 6


def test_auto_processes_support_small_machines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Automatic execution never resolves below one process."""
    monkeypatch.setattr(
        "baymorph.fitting.execution.os.cpu_count",
        lambda: 2,
    )

    assert resolve_process_count(
        "auto"
    ) == 2


def test_explicit_process_count_is_preserved() -> None:
    """Explicit process counts remain under user control."""
    assert resolve_process_count(
        24
    ) == 24


@pytest.mark.parametrize(
    "value",
    [
        0,
        -1,
    ],
)
def test_non_positive_process_count_is_rejected(
    value: int,
) -> None:
    """Worker counts must be positive."""
    with pytest.raises(
        ValueError,
        match="at least 1",
    ):
        resolve_process_count(
            value
        )


def test_boolean_process_count_is_rejected() -> None:
    """Boolean values must not be interpreted as integer worker counts."""
    with pytest.raises(
        TypeError,
        match="positive integer",
    ):
        resolve_process_count(
            True
        )


def test_unknown_process_mode_is_rejected() -> None:
    """Only the automatic process mode has symbolic meaning."""
    with pytest.raises(
        ValueError,
        match="'auto'",
    ):
        resolve_process_count(
            "many"
        )


def test_execution_defaults_are_parallel_and_visible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The canonical execution profile enables automatic parallelism."""
    monkeypatch.setattr(
        "baymorph.fitting.execution.os.cpu_count",
        lambda: 12,
    )

    execution = resolve_sampler_execution()

    assert execution.requested_processes == "auto"
    assert execution.processes == 10
    assert execution.progress is True


def test_execution_accepts_serial_quiet_mode() -> None:
    """Tests and controlled workflows can request serial quiet execution."""
    execution = resolve_sampler_execution(
        {
            "processes": 1,
            "progress": False,
        }
    )

    assert execution.processes == 1
    assert execution.progress is False


def test_unknown_execution_option_is_rejected() -> None:
    """Backend-neutral execution has a deliberately small public surface."""
    with pytest.raises(
        ValueError,
        match="Unknown sampler execution option",
    ):
        resolve_sampler_execution(
            {
                "threads": 4,
            }
        )


def test_progress_must_be_boolean() -> None:
    """Progress semantics must not depend on truthiness coercion."""
    with pytest.raises(
        TypeError,
        match="progress must be a boolean",
    ):
        resolve_sampler_execution(
            {
                "progress": 1,
            }
        )


def _square_in_worker(
    value: int,
) -> int:
    """Return a deterministic value from a worker process."""
    return value * value


def test_sampler_pool_is_serial_for_one_process() -> None:
    """Serial execution must not construct a worker pool."""
    from baymorph.fitting.execution import sampler_pool

    with sampler_pool(
        1
    ) as pool:
        assert pool is None


def test_sampler_pool_executes_in_multiple_processes() -> None:
    """The production pool must execute picklable work in child processes."""
    from baymorph.fitting.execution import sampler_pool

    with sampler_pool(
        2
    ) as pool:
        assert pool is not None

        values = pool.map(
            _square_in_worker,
            [
                1,
                2,
                3,
                4,
            ],
        )

    assert values == [
        1,
        4,
        9,
        16,
    ]
