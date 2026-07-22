"""Base interfaces and shared validation for image likelihoods."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class ImageLikelihood(ABC):
    """Abstract base class for pixel-based image likelihoods.

    A likelihood object represents a fixed observed dataset. The observed
    image and optional mask are validated once during initialization. During
    inference, only the model image changes.

    Parameters
    ----------
    data : array-like
        Observed image.
    mask : array-like of bool, optional
        Boolean mask identifying pixels excluded from likelihood evaluation.
        Masked pixels are represented by ``True`` values.

    Notes
    -----
    Subclasses implement :meth:`_log_likelihood`, which receives the model
    values corresponding only to the unmasked pixels.
    """

    def __init__(
        self,
        data: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> None:
        data_array = self._validate_image(
            data,
            name="Data",
        )

        mask_array = self._validate_mask(
            mask,
            shape=data_array.shape,
        )

        if mask_array is None:
            valid_pixels = np.ones(
                data_array.shape,
                dtype=bool,
            )
        else:
            valid_pixels = ~mask_array

        if not np.any(valid_pixels):
            raise ValueError(
                "Likelihood evaluation requires at least one unmasked pixel."
            )

        data_values = np.asarray(
            data_array[valid_pixels],
            dtype=float,
        )

        if not np.all(np.isfinite(data_values)):
            raise ValueError(
                "Data values must be finite in all evaluated pixels."
            )

        data_array.setflags(write=False)
        valid_pixels.setflags(write=False)
        data_values.setflags(write=False)

        if mask_array is not None:
            mask_array.setflags(write=False)

        self._data = data_array
        self._mask = mask_array
        self._valid_pixels = valid_pixels
        self._data_values = data_values

    @property
    def data(self) -> np.ndarray:
        """Observed image."""
        return self._data

    @property
    def mask(self) -> np.ndarray | None:
        """Boolean exclusion mask."""
        return self._mask

    @property
    def image_shape(self) -> tuple[int, ...]:
        """Shape of the observed image."""
        return self._data.shape

    @property
    def data_values(self) -> np.ndarray:
        """Observed values in unmasked pixels."""
        return self._data_values

    @property
    def n_valid_pixels(self) -> int:
        """Number of pixels included in likelihood evaluation."""
        return int(self._data_values.size)

    def __call__(
        self,
        model: np.ndarray,
    ) -> float:
        """Evaluate the log-likelihood of a model image.

        Parameters
        ----------
        model : array-like
            Model image evaluated on the same pixel grid as the observed
            image.

        Returns
        -------
        float
            Log-likelihood value.
        """
        model_values = self.prepare_model(model)

        return float(
            self._log_likelihood(model_values)
        )

    def log_likelihood(
        self,
        model: np.ndarray,
    ) -> float:
        """Evaluate the log-likelihood of a model image.

        This named method is equivalent to calling the likelihood object
        directly.

        Parameters
        ----------
        model : array-like
            Model image evaluated on the same pixel grid as the observed
            image.

        Returns
        -------
        float
            Log-likelihood value.
        """
        return self(model)

    @abstractmethod
    def _log_likelihood(
        self,
        model_values: np.ndarray,
    ) -> float:
        """Evaluate the likelihood using validated model values.

        Parameters
        ----------
        model_values : ndarray
            One-dimensional array containing model values in the unmasked
            pixels.

        Returns
        -------
        float
            Log-likelihood value.
        """

    def prepare_model(
        self,
        model: np.ndarray,
    ) -> np.ndarray:
        """Validate a model image and select evaluated pixels.

        Parameters
        ----------
        model : array-like
            Model image evaluated on the same pixel grid as the observed
            image.

        Returns
        -------
        ndarray
            One-dimensional array containing model values in unmasked pixels.

        Raises
        ------
        ValueError
            If the model shape differs from the observed image or if the
            evaluated model pixels contain non-finite values.
        """
        model_array = self._validate_image(
            model,
            name="Model",
        )

        if model_array.shape != self.image_shape:
            raise ValueError(
                "Model must have the same shape as data."
            )

        model_values = np.asarray(
            model_array[self._valid_pixels],
            dtype=float,
        )

        if not np.all(np.isfinite(model_values)):
            raise ValueError(
                "Model values must be finite in all evaluated pixels."
            )

        return model_values

    def prepare_fixed_array(
        self,
        values: np.ndarray | float,
        *,
        name: str,
        require_positive: bool = False,
        require_non_negative: bool = False,
    ) -> np.ndarray:
        """Validate a fixed scalar or image-like quantity.

        Scalars and broadcastable arrays are expanded to the observed image
        shape. Only values corresponding to unmasked pixels are returned.

        Parameters
        ----------
        values : array-like or float
            Scalar or array broadcastable to the observed image shape.
        name : str
            Human-readable quantity name used in error messages.
        require_positive : bool, default=False
            If True, require all evaluated values to be strictly positive.
        require_non_negative : bool, default=False
            If True, require all evaluated values to be non-negative.

        Returns
        -------
        ndarray
            One-dimensional array containing values in unmasked pixels.

        Raises
        ------
        ValueError
            If the quantity cannot be broadcast to the image shape, contains
            non-finite values, or violates the requested bounds.
        """
        if require_positive and require_non_negative:
            raise ValueError(
                "A quantity cannot simultaneously require positive and "
                "non-negative validation."
            )

        values_array = np.asarray(
            values,
            dtype=float,
        )

        try:
            broadcast_values = np.broadcast_to(
                values_array,
                self.image_shape,
            )
        except ValueError as error:
            raise ValueError(
                f"{name} must be scalar or broadcastable to the data shape."
            ) from error

        selected_values = np.asarray(
            broadcast_values[self._valid_pixels],
            dtype=float,
        )

        if not np.all(np.isfinite(selected_values)):
            raise ValueError(
                f"{name} values must be finite in all evaluated pixels."
            )

        if require_positive and np.any(selected_values <= 0.0):
            raise ValueError(
                f"{name} values must be positive in all evaluated pixels."
            )

        if require_non_negative and np.any(selected_values < 0.0):
            raise ValueError(
                f"{name} values must be non-negative in all evaluated pixels."
            )

        selected_values.setflags(write=False)

        return selected_values

    @staticmethod
    def _validate_image(
        image: np.ndarray,
        *,
        name: str,
    ) -> np.ndarray:
        """Convert an image to an independent floating-point NumPy array."""
        image_array = np.array(
            image,
            dtype=float,
            copy=True,
        )

        if image_array.ndim < 1:
            raise ValueError(
                f"{name} must have at least one dimension."
            )

        return image_array

    @staticmethod
    def _validate_mask(
        mask: np.ndarray | None,
        *,
        shape: tuple[int, ...],
    ) -> np.ndarray | None:
        """Validate an optional Boolean exclusion mask."""
        if mask is None:
            return None

        mask_array = np.array(
            mask,
            dtype=bool,
            copy=True,
        )

        if mask_array.shape != shape:
            raise ValueError(
                "Mask must have the same shape as data."
            )

        return mask_array