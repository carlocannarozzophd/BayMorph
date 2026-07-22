from baymorph.validation import benchmark_call


def test_benchmark_call_returns_value_and_non_negative_runtime():
    result = benchmark_call(lambda x: x + 1, 2)

    assert result.value == 3
    assert result.runtime_seconds >= 0.0
