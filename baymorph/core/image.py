"""Image containers and FITS I/O."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits


@dataclass
class ImageData:
    """Container for astronomical image data.

    Parameters
    ----------
    data : ndarray
        Two-dimensional image array.
    header : astropy.io.fits.Header, optional
        FITS header associated with the image.
    path : str, optional
        Path to the image on disk. This is informational and is not updated
        automatically when writing the image to a new file.
    """

    data: np.ndarray
    header: fits.Header | None = None
    path: str | None = None

    @classmethod
    def from_fits(cls, filename: str | Path) -> "ImageData":
        """Load an image from a FITS file.

        Parameters
        ----------
        filename : str or pathlib.Path
            Path to the FITS file.

        Returns
        -------
        ImageData
            Image container initialized from the primary HDU of the FITS file.
        """
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
        """Write the image to a FITS file.

        Parameters
        ----------
        filename : str or pathlib.Path
            Output FITS filename.
        overwrite : bool, default=True
            If True, overwrite an existing file.
        """
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
        """tuple of int : Shape of the image array as ``(ny, nx)``."""
        return self.data.shape