"""High-level BayMorph workflow runner."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .pipeline import (
    run_image_preparation,
    run_psf_candidate_selection,
    run_psf_construction,
    run_real_fit,
)

@dataclass
class BayMorphRunner:
    """High-level orchestrator for BayMorph workflows."""

    config: dict[str, Any]

    image_preparation_products: Any = None
    psf_candidate_products: Any = None
    psf_products: Any = None
    fit_result: Any = None
    profile_bands: Any = None

    @classmethod
    def from_config(cls, config):
        return cls(
            config=dict(config),
        )

    @property
    def output_dir(self):
        value = self.config.get("output_dir")

        if value is None:
            return None

        return Path(value)

    def prepare_image(self):
        """Run image preparation and store the resulting products."""
        self.image_preparation_products = run_image_preparation(
            self.config,
        )

        return self

    def select_psf_candidates(self):
        """Run PSF candidate selection and store the results."""
        self.psf_candidate_products = (
            run_psf_candidate_selection(
                self.config,
            )
        )

        return self

    def build_psf(self):
        """Run empirical PSF construction and store the products."""
        self.psf_products = run_psf_construction(
            self.config,
        )

        return self


    def fit(self):
        """Run the real-image fitting workflow and store the result."""
        self.fit_result = run_real_fit(
            self.config,
        )

        return self

    def posterior_profiles(self):
        raise NotImplementedError

    def run_all(
        self,
        *,
        include_posterior_profiles=False,
    ):
        self.prepare_image()
        self.select_psf_candidates()
        self.build_psf()
        self.fit()

        if include_posterior_profiles:
            self.posterior_profiles()

        return self