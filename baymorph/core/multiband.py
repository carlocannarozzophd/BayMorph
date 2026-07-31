"""Multi-band astronomical observations."""

from __future__ import annotations

from dataclasses import dataclass

from baymorph.core.observation import Observation


@dataclass(frozen=True)
class MultiBandObservation:
    """Collection of observations of the same target in multiple bands."""

    bands: dict[str, Observation]

    def __post_init__(self) -> None:
        if not self.bands:
            raise ValueError("MultiBandObservation requires at least one band.")

        normalized = {}

        for name, observation in self.bands.items():
            if not isinstance(name, str):
                raise TypeError("Band names must be strings.")

            if not isinstance(
                observation,
                Observation,
            ):
                raise TypeError("All values must be Observation instances.")

            normalized[str(name)] = observation

        object.__setattr__(
            self,
            "bands",
            normalized,
        )

    @property
    def filters(self) -> tuple[str, ...]:
        """Return available filter names."""
        return tuple(self.bands.keys())

    def __getitem__(
        self,
        band: str,
    ) -> Observation:
        return self.bands[band]

    def __iter__(self):
        return iter(self.bands.items())

    def __len__(self) -> int:
        return len(self.bands)
