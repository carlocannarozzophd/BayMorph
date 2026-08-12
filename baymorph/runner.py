"""High-level BayMorph workflow runner."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from baymorph.pipeline import (
    run_image_preparation,
    run_psf_candidate_selection,
    run_psf_construction,
    run_real_fit,
)
from baymorph.pipeline.config import (
    load_config,
    load_resolved_config,
    resolve_config,
)
from baymorph.pipeline.multiband_config import (
    resolve_multiband_observation_configs,
)
from baymorph.pipeline.multiband_fit import (
    run_multi_observation_fit_from_config,
)

WorkflowMode = Literal[
    "single",
    "multi_observation",
]


def _workflow_mode_from_config(
    config: Mapping[str, Any],
) -> WorkflowMode:
    """Return the workflow mode implied by a raw configuration.

    Parameters
    ----------
    config : mapping
        Raw BayMorph configuration.

    Returns
    -------
    {"single", "multi_observation"}
        Workflow mode selected from the configuration structure.

    Notes
    -----
    The presence of a top-level ``observations`` mapping identifies a
    multi-observation analysis. Otherwise the configuration is interpreted
    through the established single-observation configuration path.
    """
    if "observations" in config:
        observations = config["observations"]

        if (
            not isinstance(
                observations,
                Mapping,
            )
            or not observations
        ):
            raise ValueError(
                "Multi-observation configuration requires a non-empty "
                "'observations' mapping."
            )

        return "multi_observation"

    return "single"


class BayMorphRunner:
    """Coordinate the high-level BayMorph analysis workflows.

    The runner is the main user-facing interface for executing BayMorph
    workflows from either a configuration dictionary or a YAML configuration
    file.

    The same public runner represents both single-observation and
    multi-observation analyses. Multi-observation execution is introduced
    incrementally while preserving the established single-observation
    workflow.

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
        self.config_path: Path | None = None
        self.config_base_dir: Path | None = None

        if isinstance(
            config,
            (str, Path),
        ):
            self.config_path = Path(config)
            self.config_base_dir = (
                self.config_path.parent
            )

            raw_config = load_config(
                self.config_path
            )
            self.workflow_mode = (
                _workflow_mode_from_config(
                    raw_config
                )
            )

            if self.workflow_mode == "single":
                self.config = load_resolved_config(
                    self.config_path,
                )
            else:
                self.config = dict(
                    raw_config
                )
        else:
            raw_config = dict(config)
            self.workflow_mode = (
                _workflow_mode_from_config(
                    raw_config
                )
            )

            if self.workflow_mode == "single":
                self.config = resolve_config(
                    raw_config
                )
            else:
                self.config = raw_config

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
    def is_multi_observation(self) -> bool:
        """Return whether this runner represents multiple observations."""
        return (
            self.workflow_mode
            == "multi_observation"
        )

    def _require_single_observation(
        self,
        operation: str,
    ) -> None:
        """Reject not-yet-integrated multi-observation workflow stages."""
        if self.is_multi_observation:
            raise NotImplementedError(
                f"{operation}() is not yet exposed through BayMorphRunner "
                "for multi-observation configurations."
            )

    @property
    def output_dir(self) -> Path:
        """Return the configured output root directory."""
        self._require_single_observation(
            "output_dir"
        )

        return Path(
            self.config["outputs"]["root"]
        )

    @property
    def run_dir(self) -> Path:
        """Return the output directory for the configured object."""
        self._require_single_observation(
            "run_dir"
        )

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
        """Run image preparation for the configured observations.

        Returns
        -------
        Any
            Single-observation preparation products or a mapping from
            observation name to preparation products.
        """
        if self.is_multi_observation:
            observation_configs = (
                resolve_multiband_observation_configs(
                    self.config,
                    base_dir=self.config_base_dir,
                )
            )

            self.image_preparation_products = {
                name: run_image_preparation(
                    observation_config
                )
                for name, observation_config
                in observation_configs.items()
            }

            self._status(
                "Multi-observation image preparation completed"
            )

            return self.image_preparation_products

        self.image_preparation_products = (
            run_image_preparation(
                self.config,
            )
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
            Single-observation PSF candidates or a mapping from observation
            name to candidate products.
        """
        if self.is_multi_observation:
            observation_configs = (
                resolve_multiband_observation_configs(
                    self.config,
                    base_dir=self.config_base_dir,
                )
            )

            self.psf_candidate_products = {
                name: run_psf_candidate_selection(
                    observation_config
                )
                for name, observation_config
                in observation_configs.items()
            }

            self._status(
                "Multi-observation PSF candidates selected"
            )

            return self.psf_candidate_products

        self.psf_candidate_products = (
            run_psf_candidate_selection(
                self.config,
            )
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
        """Build empirical PSFs for the configured observations.

        Returns
        -------
        Any
            Single-observation PSF products or a mapping from observation
            name to PSF products.
        """
        if self.is_multi_observation:
            observation_configs = (
                resolve_multiband_observation_configs(
                    self.config,
                    base_dir=self.config_base_dir,
                )
            )

            self.psf_products = {
                name: run_psf_construction(
                    observation_config
                )
                for name, observation_config
                in observation_configs.items()
            }

            self._status(
                "Multi-observation empirical PSFs constructed"
            )

            return self.psf_products

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
            Fitting products for a single- or multi-observation analysis.
        """
        if self.is_multi_observation:
            config_source = (
                self.config_path
                if self.config_path is not None
                else self.config
            )

            self.fit_result = (
                run_multi_observation_fit_from_config(
                    config_source
                )
            )

            self.profile_bands = None

            self._status(
                "Multi-observation Bayesian fitting completed"
            )

            return self.fit_result

        self.fit_result = run_real_fit(
            self.config,
        )

        self.profile_bands = None

        self._status(
            "Bayesian fitting completed"
        )

        return self.fit_result

    def corner(self) -> None:
        """Generate corner-plot diagnostics."""
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
        """Generate residual-map diagnostics."""
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
        """Generate radial-profile diagnostics."""
        self._require_fit_result(
            "radial_profile"
        )

        from baymorph.pipeline import run_radial_profile_diagnostics

        self.profile_bands = (
            run_radial_profile_diagnostics(
                self.config,
                self.fit_result,
            )
        )

        self._status(
            "Radial profiles generated"
        )

        return self.profile_bands

    def diagnostics(self) -> Any:
        """Generate the diagnostics enabled by the configuration."""
        self._require_fit_result(
            "diagnostics"
        )

        diagnostics_config = self.config.get(
            "diagnostics",
            {},
        )

        if diagnostics_config is None:
            diagnostics_config = {}

        if not isinstance(
            diagnostics_config,
            Mapping,
        ):
            raise TypeError(
                "diagnostics must be a mapping."
            )

        corner_enabled = bool(
            diagnostics_config.get(
                "corner",
                True,
            )
        )

        residual_enabled = bool(
            diagnostics_config.get(
                "residual",
                True,
            )
        )

        radial_setting = diagnostics_config.get(
            "radial_profile",
            True,
        )

        if isinstance(
            radial_setting,
            Mapping,
        ):
            radial_enabled = True
        else:
            radial_enabled = bool(
                radial_setting
            )

        if corner_enabled:
            self.corner()

        if residual_enabled:
            self.residual_maps()

        if radial_enabled:
            profile_bands = self.radial_profile()
        else:
            self.profile_bands = None
            profile_bands = None

        self._status(
            "Configured diagnostics generated"
        )

        return profile_bands

    def run(
        self,
        *,
        diagnostics: bool = False,
    ) -> Any:
        """Run the configured BayMorph workflow.

        Parameters
        ----------
        diagnostics : bool, default=False
            If True, generate all fit diagnostics after fitting.

        Returns
        -------
        Any
            Fitting products for the configured analysis.
        """
        self.prepare_image()

        if self.is_multi_observation:
            observation_configs = (
                resolve_multiband_observation_configs(
                    self.config,
                    base_dir=self.config_base_dir,
                )
            )

            empirical_configs = {
                name: observation_config
                for name, observation_config
                in observation_configs.items()
                if str(
                    observation_config.get(
                        "psf",
                        {},
                    ).get(
                        "type",
                        "empirical",
                    )
                ).strip().lower()
                == "empirical"
            }

            if empirical_configs:
                self.psf_candidate_products = {
                    name: run_psf_candidate_selection(
                        observation_config
                    )
                    for name, observation_config
                    in empirical_configs.items()
                }

                self.psf_products = {
                    name: run_psf_construction(
                        observation_config
                    )
                    for name, observation_config
                    in empirical_configs.items()
                }

            result = self.fit()

            if diagnostics:
                self.diagnostics()

            return result

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
