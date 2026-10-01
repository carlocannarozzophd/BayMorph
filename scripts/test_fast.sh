#!/usr/bin/env bash
set -euo pipefail

# Fast local regression gate.
#
# Expensive scientific recovery and sampler-validation tests remain part of
# the authoritative complete suite:
#
#     pytest -q

python -m pytest -q \
    --ignore=tests/test_background_gradient_recovery.py \
    --ignore=tests/test_multi_observation_recovery.py \
    --ignore=tests/test_multi_observation_dynesty.py \
    --ignore=tests/test_sersic_point_source_recovery.py \
    --ignore=tests/test_bulge_disk_recovery.py \
    --ignore=tests/test_empirical_psf_builder_benchmark.py \
    -k "not test_dynesty_generic_fit_runs \
        and not test_gaussian_mixture_reference_is_valid \
        and not test_normal_marginal_prior_is_preserved \
        and not test_nautilus_runs_with_two_processes \
        and not test_marginalized_evidence_agrees_with_the_complete_estimator \
        and not test_cusp_on_a_pixel_centre_is_a_smooth_problem \
        and not test_marginal_likelihood_matches_brute_force_integration" \
    "$@"
