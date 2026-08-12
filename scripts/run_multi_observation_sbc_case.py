"""Run one preparation-free multi-observation SBC realization."""

from __future__ import annotations

import argparse
from pathlib import Path

from baymorph.validation import (
    MultiObservationGroundTruth,
    ObservationTruth,
    PreparationFreeMultiObservationSBCCampaign,
)


def build_reference_truth() -> MultiObservationGroundTruth:
    """Return the fixed observation geometry used by the SBC campaign."""
    return MultiObservationGroundTruth(
        model_name="single_sersic",
        parameters={
            "flux__g": 900.0,
            "flux__r": 1200.0,
            "re_arcsec": 1.2,
            "n": 2.0,
            "q": 0.65,
            "position_angle": 0.0,
            "ra": 150.0,
            "dec": 30.0,
            "sky__g": 0.0,
            "sky__r": 0.0,
        },
        observations={
            "g": ObservationTruth(
                image_shape=(61, 61),
                pixel_scale=0.20,
                psf_metadata={"fwhm_pixels": 2.0},
                noise_metadata={"sigma": 8.0},
                metadata={
                    "rotation_deg": 0.0,
                    "crpix": (31.0, 31.0),
                    "seed": 11,
                },
            ),
            "r": ObservationTruth(
                image_shape=(47, 73),
                pixel_scale=0.35,
                psf_metadata={"fwhm_pixels": 2.8},
                noise_metadata={"sigma": 11.0},
                metadata={
                    "rotation_deg": 32.0,
                    "crpix": (38.0, 24.0),
                    "seed": 22,
                },
            ),
        },
        random_seed=42,
    )


def build_sampler(pool: int) -> dict:
    """Return the pocoMC configuration used by the definitive SBC campaign."""
    init = {
        "n_effective": 512,
        "n_active": 256,
        "sample": "tpcn",
        "random_state": 42,
    }

    if pool > 1:
        init["pool"] = pool

    return {
        "name": "pocomc",
        "init": init,
        "run": {
            "n_total": 4096,
            "n_evidence": 4096,
        },
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "index",
        type=int,
        help="Zero-based SBC realization index.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("validation/multi_observation_sbc_definitive"),
        help="Campaign output root.",
    )
    parser.add_argument(
        "--pool",
        type=int,
        default=10,
        help="Number of pocoMC worker processes.",
    )
    return parser.parse_args()


def main() -> None:
    """Run one deterministic SBC realization."""
    args = parse_args()

    if args.index < 0 or args.index >= 100:
        raise ValueError("SBC index must satisfy 0 <= index < 100.")

    if args.pool <= 0:
        raise ValueError("pocoMC pool size must be positive.")

    campaign = PreparationFreeMultiObservationSBCCampaign(
        root=args.root,
        truth=build_reference_truth(),
        sampler=build_sampler(args.pool),
        n_cases=100,
    )

    result_path = campaign._result_path(args.index)

    if result_path.exists():
        print(
            f"{campaign._case_id(args.index)} already completed: "
            f"{result_path}"
        )
        return

    print("=" * 72)
    print(f"Running {campaign._case_id(args.index)}")
    print(f"pocoMC pool: {args.pool}")
    print(f"Output root: {args.root}")
    print("=" * 72)

    result = campaign.run_case(args.index)

    metadata = result.metadata or {}

    print("=" * 72)
    print(f"Completed {result.case_id}")
    print(f"Likelihood calls: {result.likelihood_evaluations}")
    print(f"Runtime [s]: {result.runtime_seconds}")
    print(
        "ESS:",
        metadata.get("effective_sample_size"),
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
