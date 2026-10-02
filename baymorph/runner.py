"""High-level BayMorph workflow runner."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from baymorph.likelihood.factory import normalize_likelihood_type
from baymorph.pipeline import (
    run_image_preparation,
    run_psf_candidate_selection,
    run_psf_construction,
    run_real_fit,
    run_simulation_from_config,
)
from baymorph.pipeline.config import (
    load_config,
    load_resolved_config,
    resolve_config,
    resolve_likelihood_config,
)
from baymorph.pipeline.image_preparation import (
    image_preparation_is_complete,
    load_image_preparation_products,
)
from baymorph.pipeline.multi_observation_config import (
    resolve_multi_observation_observation_configs,
)
from baymorph.pipeline.multi_observation_fit import (
    run_multi_observation_fit_from_config,
)
from baymorph.pipeline.psf_candidates import (
    load_psf_candidates,
    psf_candidates_exist,
)
from baymorph.pipeline.psf_construction import (
    empirical_psf_exists,
    load_empirical_psf,
)
from baymorph.pipeline.noise_calibration import (
    build_correlated_noise_calibration,
    correlated_noise_calibration_path,
    reject_correlated_noise_calibration_settings,
)
from baymorph.pipeline.vecchia_factorization import (
    DEFAULT_MIN_CONDITIONAL_VARIANCE,
    PRODUCTION_VECCHIA_NEIGHBORS,
    ensure_vecchia_factorization_artifact,
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


def _locate_config_path(
    config: str | Path,
) -> Path:
    """Locate a project-root-relative configuration path.

    Relative paths are checked from the current directory upward through its
    parents. Absolute paths are preserved. If no file is found, the path
    relative to the current directory is returned so the normal configuration
    loader can report the missing file.
    """
    requested = Path(
        config
    ).expanduser()

    if requested.is_absolute():
        return requested.resolve()

    search_start = Path.cwd().resolve()

    for directory in (
        search_start,
        *search_start.parents,
    ):
        candidate = (
            directory
            / requested
        ).resolve()

        if candidate.is_file():
            return candidate

    return (
        search_start
        / requested
    ).resolve()


def _observation_psf_type(
    config: Mapping[str, Any],
) -> str:
    """Return the declared PSF type of one observation configuration."""
    return str(
        config.get(
            "psf",
            {},
        ).get(
            "type",
            "empirical",
        )
    ).strip().lower()


def _require_empirical_psf(
    config: Mapping[str, Any],
    *,
    action: str,
) -> None:
    """Raise a clear error when PSF construction is requested for a non-empirical PSF."""
    psf_type = _observation_psf_type(
        config
    )

    if psf_type != "empirical":
        raise ValueError(
            f"{action} requires psf.type='empirical'; this configuration "
            f"declares psf.type={psf_type!r}. Analytic PSFs (for example "
            "'gaussian') need no construction step."
        )


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
        self.project_root: Path | None = None

        if isinstance(
            config,
            (str, Path),
        ):
            self.config_path = _locate_config_path(
                config
            )
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

        if (
            self.workflow_mode == "single"
            and self.config_path is not None
            and self.config_path.parent.name == "configs"
        ):
            self.project_root = (
                self.config_path.parent.parent
            )
            self._anchor_project_paths()

        self.verbose = bool(verbose)

        self.simulation_observation: Any = None
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

    def _anchor_project_paths(self) -> None:
        """Anchor primary relative paths to the discovered project root."""
        if self.project_root is None:
            return

        for section_name, key in (
            (
                "outputs",
                "root",
            ),
            (
                "data",
                "image_path",
            ),
        ):
            section = self.config.get(
                section_name
            )

            if not isinstance(
                section,
                dict,
            ):
                continue

            value = section.get(
                key
            )

            if value is None:
                continue

            path_value = Path(
                value
            ).expanduser()

            if path_value.is_absolute():
                continue

            section[key] = str(
                (
                    self.project_root
                    / path_value
                ).resolve()
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
    def simulation(self) -> Any:
        """Return the generated synthetic observation, when available."""
        return self.simulation_observation

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

    def simulate(self) -> Any:
        """Generate and persist the configured synthetic observation.

        Returns
        -------
        SyntheticObservation
            Generated intrinsic, convolved, sky-added, and noisy images.

        Raises
        ------
        ValueError
            If the configuration does not define a simulation.
        """
        self._require_single_observation(
            "simulate"
        )

        if "simulation" not in self.config:
            raise ValueError(
                "simulate() requires a top-level 'simulation' "
                "configuration."
            )

        self.simulation_observation = (
            run_simulation_from_config(
                self.config
            )
        )

        self._status(
            "Synthetic observation generated"
        )

        return self.simulation_observation

    @staticmethod
    def _reuses_products(
        config: Mapping[str, Any],
    ) -> bool:
        """Return whether accepted products may be reused for this config.

        A configuration without a resolved ``outputs`` section cannot address
        persisted products, so its stages always rebuild.
        """
        outputs = config.get(
            "outputs"
        )

        if not isinstance(
            outputs,
            Mapping,
        ):
            return False

        return not bool(
            outputs.get(
                "overwrite",
                False,
            )
        )

    def _stage_status(
        self,
        message: str,
        *,
        reused: bool,
        observation_name: str | None = None,
    ) -> None:
        """Report a completed stage, naming reuse of accepted products."""
        suffix = (
            ""
            if observation_name is None
            else f" for observation '{observation_name}'"
        )

        if reused:
            self._status(
                f"Accepted {message} reused" + suffix
            )
        else:
            self._status(
                message[0].upper()
                + message[1:]
                + " completed"
                + suffix
            )

    def _prepare_one_observation(
        self,
        config: Mapping[str, Any],
    ) -> tuple[Any, bool]:
        """Prepare one observation, reusing accepted products when present.

        Reuse never bypasses validation: the loader accepts persisted products
        only when their immutable inputs still match this configuration, and
        fails closed otherwise.
        """
        if self._reuses_products(
            config
        ) and image_preparation_is_complete(
            config
        ):
            return (
                load_image_preparation_products(
                    config
                ),
                True,
            )

        return (
            run_image_preparation(
                config
            ),
            False,
        )

    def prepare_image(self) -> Any:
        """Run image preparation for the configured observations.

        A completed preparation whose identity still matches its inputs is
        loaded instead of rebuilt, so this stage can be repeated safely.
        Setting ``outputs.overwrite`` rebuilds it.

        Returns
        -------
        Any
            Single-observation preparation products or a mapping from
            observation name to preparation products.
        """
        if self.is_multi_observation:
            observation_configs = (
                resolve_multi_observation_observation_configs(
                    self.config,
                    base_dir=self.config_base_dir,
                )
            )

            products = {}

            for name, observation_config in (
                observation_configs.items()
            ):
                (
                    products[name],
                    reused,
                ) = self._prepare_one_observation(
                    observation_config
                )

                self._stage_status(
                    "image preparation",
                    reused=reused,
                    observation_name=name,
                )

            self.image_preparation_products = products

            return self.image_preparation_products

        (
            self.image_preparation_products,
            reused,
        ) = self._prepare_one_observation(
            self.config
        )

        self._stage_status(
            "image preparation",
            reused=reused,
        )

        return self.image_preparation_products

    def _select_one_psf_candidate_set(
        self,
        config: Mapping[str, Any],
    ) -> tuple[Any, bool]:
        """Select candidates, reusing an accepted candidate set when present.

        Candidates are the provenance of an accepted PSF, so they are reused
        rather than rewritten.
        """
        if self._reuses_products(
            config
        ) and psf_candidates_exist(
            config
        ):
            return (
                load_psf_candidates(
                    config
                ),
                True,
            )

        return (
            run_psf_candidate_selection(
                config
            ),
            False,
        )

    def select_psf_candidates(self) -> Any:
        """Run empirical-PSF candidate selection.

        An accepted candidate set is loaded instead of reselected, so this
        stage can be repeated safely. Setting ``outputs.overwrite`` rebuilds
        it.

        Returns
        -------
        Any
            Single-observation PSF candidates or a mapping from observation
            name to candidate products.
        """
        if self.is_multi_observation:
            observation_configs = (
                resolve_multi_observation_observation_configs(
                    self.config,
                    base_dir=self.config_base_dir,
                )
            )

            empirical_configs = {
                name: observation_config
                for name, observation_config
                in observation_configs.items()
                if _observation_psf_type(
                    observation_config
                )
                == "empirical"
            }

            candidates = {}

            for name, observation_config in (
                empirical_configs.items()
            ):
                (
                    candidates[name],
                    reused,
                ) = self._select_one_psf_candidate_set(
                    observation_config
                )

                self._stage_status(
                    "PSF candidate selection",
                    reused=reused,
                    observation_name=name,
                )

            self.psf_candidate_products = candidates

            return self.psf_candidate_products

        _require_empirical_psf(
            self.config,
            action="PSF candidate selection",
        )

        (
            self.psf_candidate_products,
            reused,
        ) = self._select_one_psf_candidate_set(
            self.config
        )

        try:
            n_candidates = len(
                self.psf_candidate_products
            )
        except TypeError:
            self._stage_status(
                "PSF candidate selection",
                reused=reused,
            )
        else:
            selection = (
                "reused" if reused else "selected"
            )

            self._status(
                f"{n_candidates} PSF candidates {selection}"
            )

        return self.psf_candidate_products

    def _build_one_psf(
        self,
        config: Mapping[str, Any],
    ) -> tuple[Any, bool]:
        """Build one PSF, reusing an accepted kernel when present.

        The loader validates the persisted kernel and its provenance before
        returning it, and fails closed otherwise. Both paths return the PSF
        image itself.
        """
        if self._reuses_products(
            config
        ) and empirical_psf_exists(
            config
        ):
            return (
                load_empirical_psf(
                    config
                ).kernel_array,
                True,
            )

        return (
            run_psf_construction(
                config
            ),
            False,
        )

    def build_psf(self) -> Any:
        """Build empirical PSFs for the configured observations.

        An accepted PSF is loaded instead of rebuilt, so this stage can be
        repeated safely. Setting ``outputs.overwrite`` rebuilds it.

        Returns
        -------
        Any
            Single-observation PSF products or a mapping from observation
            name to PSF products.
        """
        if self.is_multi_observation:
            observation_configs = (
                resolve_multi_observation_observation_configs(
                    self.config,
                    base_dir=self.config_base_dir,
                )
            )

            empirical_configs = {
                name: observation_config
                for name, observation_config
                in observation_configs.items()
                if _observation_psf_type(
                    observation_config
                )
                == "empirical"
            }

            psfs = {}

            for name, observation_config in (
                empirical_configs.items()
            ):
                (
                    psfs[name],
                    reused,
                ) = self._build_one_psf(
                    observation_config
                )

                self._stage_status(
                    "empirical PSF",
                    reused=reused,
                    observation_name=name,
                )

            self.psf_products = psfs

            return self.psf_products

        _require_empirical_psf(
            self.config,
            action="PSF construction",
        )

        (
            self.psf_products,
            reused,
        ) = self._build_one_psf(
            self.config
        )

        self._stage_status(
            "empirical PSF",
            reused=reused,
        )

        return self.psf_products

    def _ensure_observation_noise_calibration(
        self,
        config: Mapping[str, Any],
        *,
        observation_name: str | None = None,
    ) -> None:
        """Build or validate immutable correlated-noise fit artifacts."""
        likelihood_config = resolve_likelihood_config(
            config["fit"].get("likelihood")
        )
        if isinstance(likelihood_config, str):
            likelihood_type = likelihood_config
        elif isinstance(likelihood_config, Mapping):
            likelihood_type = likelihood_config.get(
                "type",
                "correlated_gaussian",
            )
        else:
            raise TypeError(
                "fit.likelihood must be a string or mapping."
            )

        if normalize_likelihood_type(likelihood_type) != "correlated_gaussian":
            return
        if not isinstance(likelihood_config, Mapping):
            raise TypeError(
                "Correlated Gaussian production fitting requires a "
                "fit.likelihood mapping."
            )

        reject_correlated_noise_calibration_settings(config)
        calibration_path = correlated_noise_calibration_path(config)
        if not calibration_path.is_file():
            build_correlated_noise_calibration(
                config,
                save=True,
                overwrite=False,
            )

        suffix = (
            ""
            if observation_name is None
            else f" for observation '{observation_name}'"
        )
        self._status(
            "Ready: correlated-noise calibration" + suffix
        )

        factorization_config = likelihood_config.get("factorization")
        if factorization_config is None:
            return
        if not isinstance(factorization_config, Mapping):
            raise TypeError(
                "fit.likelihood.factorization must be a mapping."
            )

        factorization_type = (
            str(factorization_config.get("type", ""))
            .strip()
            .lower()
            .replace("-", "_")
        )
        if factorization_type != "nearest_neighbor":
            raise ValueError(
                "Correlated Gaussian production fitting supports only "
                "factorization.type='nearest_neighbor'."
            )

        ensure_vecchia_factorization_artifact(
            config,
            max_neighbors=int(
                factorization_config.get(
                    "max_neighbors",
                    PRODUCTION_VECCHIA_NEIGHBORS,
                )
            ),
            min_conditional_variance=float(
                factorization_config.get(
                    "min_conditional_variance",
                    DEFAULT_MIN_CONDITIONAL_VARIANCE,
                )
            ),
        )
        self._status(
            "Ready: nearest-neighbour factorization" + suffix
        )

    def _ensure_multi_observation_noise_calibrations(
        self,
    ) -> None:
        """Build required noise calibrations for all observations."""
        observation_configs = (
            resolve_multi_observation_observation_configs(
                self.config,
                base_dir=self.config_base_dir,
            )
        )

        for observation_name, observation_config in (
            observation_configs.items()
        ):
            self._ensure_observation_noise_calibration(
                observation_config,
                observation_name=observation_name,
            )

    def fit(self) -> Any:
        """Run Bayesian model fitting.

        Returns
        -------
        Any
            Fitting products for a single- or multi-observation analysis.
        """
        if self.is_multi_observation:
            self._ensure_multi_observation_noise_calibrations()

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

        self._ensure_observation_noise_calibration(
            self.config
        )

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
        if (
            not self.is_multi_observation
            and "simulation" in self.config
        ):
            self.simulate()

        self.prepare_image()

        if self.is_multi_observation:
            self.select_psf_candidates()
            self.build_psf()

            result = self.fit()

            if diagnostics:
                self.diagnostics()

            return result

        if _observation_psf_type(self.config) == "empirical":
            self.select_psf_candidates()
            self.build_psf()

        result = self.fit()

        if diagnostics:
            self.diagnostics()

        return result
