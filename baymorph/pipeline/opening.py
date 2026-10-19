"""Opening of the image of an observation.

Every stage of the pipeline that needs the image of an observation opens it
through :func:`open_observation`, so every stage reads the same HDU and the
array that one stage sees is the array that the others see. The array is
read-only: a stage that changes pixel values works on its own copy.

A large file is read once per process and reused while it is unchanged, so a
mosaic of several gigabytes is not read again at every stage. A small file is
read at every opening, as reading it costs less than tracking it.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from baymorph.core.header_metadata import read_header
from baymorph.core.observation import Observation, ObservationMetadata

REUSE_MIN_BYTES = 16 * 1024**2
"""Size from which a file is read once per process and then reused."""

MAX_OPEN_OBSERVATIONS = 4
"""Number of large observations kept for reuse within one process."""

_OPENED: OrderedDict[tuple, Observation] = OrderedDict()


def file_state(path: Path) -> tuple[int, int, int, int, int]:
    """Return the state that changes whenever the content of a file changes.

    Parameters
    ----------
    path : pathlib.Path
        Existing file.

    Returns
    -------
    tuple of int
        Device, inode, size, and the times of the last modification and of
        the last change of status, in nanoseconds. The second time is set by
        the system on every write and cannot be restored by copying tools,
        so a file replaced by another with the same size and modification
        time still changes state.
    """
    status = path.stat()

    return (
        status.st_dev,
        status.st_ino,
        status.st_size,
        status.st_mtime_ns,
        status.st_ctime_ns,
    )


def open_observation(
    path: str | Path,
    *,
    hdu: int | str | None = None,
    declarations: Mapping[str, Any] | None = None,
) -> Observation:
    """Open the image of an observation.

    A file of at least :data:`REUSE_MIN_BYTES` is read once per process and
    reused while its state, :func:`file_state`, is unchanged; a smaller file
    is read at every call. The image array is read-only.

    With declarations, the metadata of the observation come from the reading
    of its header with them (AD-053); the image is the same.

    Parameters
    ----------
    path : str or pathlib.Path
        FITS file of the observation.
    hdu : int or str, optional
        Index or ``EXTNAME`` of the HDU that holds the image, the
        ``data.hdu`` of a configuration. See
        :meth:`baymorph.core.image.ImageData.from_fits` for the default.
    declarations : mapping, optional
        Values of ``data.calibration``, which take the place of the readings
        of the header.

    Returns
    -------
    Observation
        Image of the selected HDU, with its header and metadata.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the declarations are not valid.
    """
    resolved = Path(path).expanduser().resolve()

    if not resolved.is_file():
        raise FileNotFoundError(f"FITS file not found: {resolved}")

    state = file_state(resolved)
    reusable = state[2] >= REUSE_MIN_BYTES
    key = (str(resolved), hdu, *state)

    if reusable and key in _OPENED:
        _OPENED.move_to_end(key)
        observation = _OPENED[key]
    else:
        observation = Observation.from_fits(resolved, hdu=hdu)
        observation.image.data.flags.writeable = False

        if reusable:
            _OPENED[key] = observation

            while len(_OPENED) > MAX_OPEN_OBSERVATIONS:
                _OPENED.popitem(last=False)

    if not declarations:
        return observation

    image = observation.image
    reading = read_header(
        image.header, primary_header=image.primary_header, declarations=declarations
    )

    return Observation(
        image=image,
        metadata=ObservationMetadata.from_header(
            image.header,
            image_shape=image.shape,
            primary_header=image.primary_header,
            reading=reading,
        ),
    )


def forget_opened_observations() -> None:
    """Release every observation kept for reuse.

    The next call of :func:`open_observation` reads its file again.
    """
    _OPENED.clear()


__all__ = [
    "MAX_OPEN_OBSERVATIONS",
    "REUSE_MIN_BYTES",
    "file_state",
    "forget_opened_observations",
    "open_observation",
]
