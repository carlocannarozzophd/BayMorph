"""High-level BayMorph workflow runner."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from baymorph.pipeline import (
    run_image_preparation,
    run_psf_candidate_selection,
    run_psf_construction,
    run_real_fit,
)
from baymorph.pipeline.config import (
    load_resolved_config,
    resolve_config,
)


class BayMorphRunner:
    """Coordinate the high-level BayMorph analysis workflows.

    The runner is the main user-facing interface for executing BayMorph
    workflows from either a configuration dictionary or a resolved YAML
    configuration file.

    Parameters
    ----------
    config : str, Path, or dict of str to Any
        BayMorph configuration file or configuration dictionary.
    verbose : bool, default=True
        If True, print concise workflow status messages.
    """

    def __init__(
        self,
        config: str | Path | dict[str, Any],
        *,
        verbose: bool = True,
    ) -> None:
        if isinstance(config, (str, Path)):
            self.config = load_resolved_config(
                config,
            )
        else:
            self.config = resolve_config(
                dict(config),
            )

        self.verbose = bool(verbose)

        self.image_preparation_products: Any = None
        self.psf_candidate_products: Any = None
        self.psf_products: Any = None
        self.fit_result: Any = None
        self.profile_bands: Any = None

    @classmethod
    def from_config(
        cls,
        config: str | Path | dict[str, Any],
        *,
        verbose: bool = True,
    ) -> BayMorphRunner:
        """Create a runner from a configuration source.

        Parameters
        ----------
        config : str, Path, or dict of str to Any
            BayMorph configuration file or configuration dictionary.
        verbose : bool, default=True
            If True, print concise workflow status messages.

        Returns
        -------
        BayMorphRunner
            Initialized workflow runner.
        """
        return cls(
            config,
            verbose=verbose,
        )

    @property
    def output_dir(self) -> Path:
        """Return the configured output root directory."""
        return Path(
            self.config["outputs"]["root"]
        )

    @property
    def run_dir(self) -> Path:
        """Return the output directory for the configured object."""
        return (
            self.output_dir
            / str(self.config["object"]["name"])
        )

    @property
    def image(self) -> Any:
        """Return the image-preparation products."""
        return self.image_preparation_products

    @property
    def psf_candidates(self) -> Any:
        """Return the selected PSF candidates."""
        return self.psf_candidate_products

    @property
    def psf(self) -> Any:
        """Return the constructed PSF products."""
        return self.psf_products

    @property
    def results(self) -> Any:
        """Return the Bayesian fitting results."""
        return self.fit_result

    def _status(
        self,
        message: str,
    ) -> None:
        """Print a completed-step status message when verbosity is enabled."""
        if self.verbose:
            print(
                f"✓ {message}"
            )

    def _require_fit_result(
        self,
        diagnostic_name: str,
    ) -> None:
        """Require a completed fit before running a diagnostic."""
        if self.fit_result is None:
            raise RuntimeError(
                f"Run fit() before {diagnostic_name}()."
            )

    def prepare_image(self) -> Any:
        """Run image preparation.

        Returns
        -------
        Any
            Image-preparation products.
        """
        self.image_preparation_products = run_image_preparation(
            self.config,
        )

        self._status(
            "Image preparation completed"
        )

        return self.image_preparation_products

    def select_psf_candidates(self) -> Any:
        """Run empirical-PSF candidate selection.

        Returns
        -------
        Any
            Selected PSF candidates.
        """
        self.psf_candidate_products = run_psf_candidate_selection(
            self.config,
        )

        try:
            n_candidates = len(
                self.psf_candidate_products
            )
        except TypeError:
            self._status(
                "PSF candidates selected"
            )
        else:
            self._status(
                f"{n_candidates} PSF candidates selected"
            )

        return self.psf_candidate_products

    def build_psf(self) -> Any:
        """Build the empirical PSF.

        Returns
        -------
        Any
            Empirical PSF products.
        """
        self.psf_products = run_psf_construction(
            self.config,
        )

        self._status(
            "Empirical PSF constructed"
        )

        return self.psf_products

    def fit(self) -> Any:
        """Run Bayesian model fitting.

        Returns
        -------
        Any
            Fitting products for one model or a mapping of model products.
        """
        self.fit_result = run_real_fit(
            self.config,
        )

        self.profile_bands = None

        self._status(
            "Bayesian fitting completed"
        )

        return self.fit_result

    def corner(self) -> None:
        """Generate corner-plot diagnostics.

        Raises
        ------
        RuntimeError
            If fitting has not been run.
        """
        self._require_fit_result(
            "corner"
        )

        from baymorph.pipeline import run_corner_diagnostics

        run_corner_diagnostics(
            self.config,
            self.fit_result,
        )

        self._status(
            "Corner plots generated"
        )

    def residual_maps(self) -> None:
        """Generate residual-map diagnostics.

        Raises
        ------
        RuntimeError
            If fitting has not been run.
        """
        self._require_fit_result(
            "residual_maps"
        )

        from baymorph.pipeline import run_residual_map_diagnostics

        run_residual_map_diagnostics(
            self.config,
            self.fit_result,
        )

        self._status(
            "Residual maps generated"
        )

    def radial_profile(self) -> Any:
        """Generate radial-profile diagnostics.

        Returns
        -------
        Any
            Posterior radial-profile bands when available.

        Raises
        ------
        RuntimeError
            If fitting has not been run.
        """
        self._require_fit_result(
            "radial_profile"
        )

        from baymorph.pipeline import run_radial_profile_diagnostics

        self.profile_bands = run_radial_profile_diagnostics(
            self.config,
            self.fit_result,
        )

        self._status(
            "Radial profiles generated"
        )

        return self.profile_bands

    def diagnostics(self) -> Any:
        """Generate all available fit diagnostics.

        Returns
        -------
        Any
            Posterior radial-profile bands when available.

        Raises
        ------
        RuntimeError
            If fitting has not been run.
        """
        self._require_fit_result(
            "diagnostics"
        )

        self.corner()
        self.residual_maps()
        profile_bands = self.radial_profile()

        self._status(
            "All diagnostics generated"
        )

        return profile_bands

    def run(
        self,
        *,
        diagnostics: bool = False,
    ) -> Any:
        """Run the configured BayMorph workflow.

        Image preparation is always executed. Empirical-PSF candidate
        selection and construction are executed only when the configured PSF
        type is ``empirical``. Bayesian fitting is then performed.

        Parameters
        ----------
        diagnostics : bool, default=False
            If True, generate all fit diagnostics after fitting.

        Returns
        -------
        Any
            Fitting products for one model or a mapping of model products.
        """
        self.prepare_image()

        psf_type = str(
            self.config.get(
                "psf",
                {},
            ).get(
                "type",
                "empirical",
            )
        ).strip().lower()

        if psf_type == "empirical":
            self.select_psf_candidates()
            self.build_psf()

        result = self.fit()

        if diagnostics:
            self.diagnostics()

        return result