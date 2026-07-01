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

    # ------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------

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
    
    @property
    def image(self):
        """Image preparation products."""
        return self.image_preparation_products


    @property
    def psf_candidates(self):
        """Selected PSF candidates."""
        return self.psf_candidate_products


    @property
    def psf(self):
        """Empirical PSF."""
        return self.psf_products


    @property
    def results(self):
        """Bayesian fitting results."""
        return self.fit_result

    # ------------------------------------------------------------
    # Console utilities
    # ------------------------------------------------------------

    @staticmethod
    def _separator():
        print("────────────────────────────────────────────────────")

    @staticmethod
    def _status(message: str):
        print(f"✓ {message}")

    @staticmethod
    def _info(message: str):
        print(f"• {message}")

    # ------------------------------------------------------------
    # Pipeline steps
    # ------------------------------------------------------------

    def prepare_image(
        self,
        *,
        return_result: bool = False,
    ):
        """Run image preparation."""

        self.image_preparation_products = run_image_preparation(
            self.config,
        )

        self._status("Image preparation completed")

        if return_result:
            return self.image_preparation_products


    def select_psf_candidates(
        self,
        *,
        return_result: bool = False,
    ):
        """Run PSF candidate selection."""

        self.psf_candidate_products = (
            run_psf_candidate_selection(
                self.config,
            )
        )

        try:
            n = len(self.psf_candidate_products)
            self._status(f"{n} PSF candidates selected")
        except Exception:
            self._status("PSF candidates selected")

        if return_result:
            return self.psf_candidate_products


    def build_psf(
        self,
        *,
        return_result: bool = False,
    ):
        """Build the empirical PSF."""

        self.psf_products = run_psf_construction(
            self.config,
        )

        self._status("Empirical PSF constructed")

        if return_result:
            return self.psf_products


    def fit(
        self,
        *,
        return_result: bool = False,
    ):
        """Run Bayesian fitting."""

        self.fit_result = run_real_fit(
            self.config,
        )

        self._status("Bayesian fitting completed")

        if return_result:
            return self.fit_result


    def corner(self):
        """Generate corner plots."""

        if self.fit_result is None:
            raise RuntimeError(
                "Run fit() before corner()."
            )

        from baymorph.pipeline import (
            run_corner_diagnostics,
        )

        run_corner_diagnostics(
            self.config,
            self.fit_result,
        )

        self._status(
            "Corner plots generated"
        )