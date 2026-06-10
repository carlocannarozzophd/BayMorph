"""Image containers and FITS I/O."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits


@dataclass
class ImageData:
    """Container for astronomical image data."""

    data: np.ndarray
    header: fits.Header | None = None
    path: str | None = None

    @classmethod
    def from_fits(cls, filename: str | Path) -> "ImageData":
        """Load image data from a FITS file."""
        filename = Path(filename)

        with fits.open(filename) as hdul:
            data = hdul[0].data.astype(float)
            header = hdul[0].header.copy()

        return cls(
            data=data,
            header=header,
            path=str(filename),
        )

    def to_fits(
        self,
        filename: str | Path,
        overwrite: bool = True,
    ) -> None:
        """Write image data to a FITS file."""
        filename = Path(filename)

        hdu = fits.PrimaryHDU(
            data=self.data,
            header=self.header,
        )

        hdu.writeto(
            filename,
            overwrite=overwrite,
        )

    @property
    def shape(self) -> tuple[int, int]:
        """Image shape."""
        return self.data.shape