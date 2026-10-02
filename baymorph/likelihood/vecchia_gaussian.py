"""Normalized nearest-neighbour correlated Gaussian image likelihood."""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree

from baymorph.likelihood.base import ImageLikelihood


VECCHIA_FACTORIZATION_SCHEMA = "baymorph.vecchia_factorization"
VECCHIA_FACTORIZATION_VERSION = 1
VECCHIA_ALGORITHM = "nearest_neighbor_conditional_gaussian"
VECCHIA_ORDERING = "row_major"
VECCHIA_NEIGHBOR_POLICY = "global_nearest_previous"
VECCHIA_DISTANCE_METRIC = "squared_euclidean_pixel"
VECCHIA_TIE_BREAK = "descending_predecessor_c_index"
_COEFFICIENT_CACHE_SIZE = 4096


def _array_sha256(array: np.ndarray) -> str:
    """Return the canonical BayMorph SHA-256 fingerprint of an array."""
    values = np.ascontiguousarray(np.asarray(array))
    digest = hashlib.sha256()
    digest.update(values.dtype.str.encode("ascii"))
    digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
    digest.update(values.view(np.uint8).tobytes())
    return digest.hexdigest()


def _readonly(
    values: np.ndarray,
    dtype: np.dtype[Any] | type,
) -> np.ndarray:
    """Return immutable owned data or preserve an authenticated read-only mmap."""
    expected = np.dtype(dtype)
    source = np.asarray(values)

    if source.dtype != expected:
        raise TypeError(
            f"Factorization array must have dtype {expected}."
        )

    if not source.flags.c_contiguous:
        raise ValueError(
            "Factorization arrays must be C-contiguous."
        )

    preserve_mmap = (
        isinstance(values, np.memmap)
        and values.mode == "r"
        and not values.flags.writeable
    )

    if preserve_mmap:
        output = source
    else:
        output = np.frombuffer(
            source.tobytes(order="C"),
            dtype=expected,
        ).reshape(source.shape)

    output.setflags(write=False)
    return output


def _mask_array(mask: np.ndarray) -> np.ndarray:
    values = np.asarray(mask)

    if values.ndim != 2:
        raise ValueError(
            "Vecchia factorization requires a two-dimensional mask."
        )

    if values.dtype.kind != "b":
        raise TypeError(
            "Vecchia factorization mask must have boolean dtype."
        )

    if not np.any(~values):
        raise ValueError(
            "Vecchia factorization requires an unmasked pixel."
        )

    output = np.array(
        values,
        dtype=bool,
        order="C",
        copy=True,
    )
    output.setflags(write=False)
    return output


def _kernel_array(
    kernel: np.ndarray,
) -> np.ndarray:
    raw = np.asarray(kernel)

    if np.iscomplexobj(raw):
        raise TypeError(
            "correlation_kernel must be real-valued."
        )

    values = np.asarray(raw, dtype=float)

    if values.ndim != 2:
        raise ValueError(
            "correlation_kernel must be two-dimensional."
        )

    if any(
        size < 1 or size % 2 == 0
        for size in values.shape
    ):
        raise ValueError(
            "correlation_kernel dimensions must be positive and odd."
        )

    if not np.all(np.isfinite(values)):
        raise ValueError(
            "correlation_kernel values must be finite."
        )

    center = tuple(
        size // 2
        for size in values.shape
    )

    if values[center] != 1.0:
        raise ValueError(
            "correlation_kernel must have exactly unit "
            "zero-lag correlation."
        )

    if not np.array_equal(
        values,
        values[::-1, ::-1],
    ):
        raise ValueError(
            "correlation_kernel must satisfy bitwise "
            "covariance symmetry."
        )

    output = np.array(
        values,
        dtype=np.float64,
        order="C",
        copy=True,
    )
    output.setflags(write=False)
    return output


def _positive_integer(
    value: int,
    *,
    name: str,
    maximum: int,
) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
    ):
        raise TypeError(
            f"{name} must be an integer."
        )

    result = int(value)

    if result < 1 or result > maximum:
        raise ValueError(
            f"{name} must lie between 1 and {maximum}."
        )

    return result


def _positive_fraction(
    value: float,
    *,
    name: str,
) -> float:
    result = float(value)

    if (
        not np.isfinite(result)
        or result <= 0.0
        or result >= 1.0
    ):
        raise ValueError(
            f"{name} must be finite and lie between 0 and 1."
        )

    return result


@dataclass(frozen=True)
class VecchiaFactorization:
    """Immutable CSR factorization on one exact row-major fitting domain."""

    image_shape: tuple[int, int]
    valid_flat_indices: np.ndarray
    neighbor_indptr: np.ndarray
    neighbor_indices: np.ndarray
    weights: np.ndarray
    conditional_variances: np.ndarray
    max_neighbors: int
    correlation_logdet: float
    correlation_kernel_sha256: str
    valid_mask_sha256: str
    n_unique_patterns: int
    min_conditional_variance: float
    schema: str = VECCHIA_FACTORIZATION_SCHEMA
    version: int = VECCHIA_FACTORIZATION_VERSION
    approximation: str = VECCHIA_ALGORITHM
    ordering: str = VECCHIA_ORDERING

    def __post_init__(self) -> None:
        if (
            not isinstance(
                self.image_shape,
                (tuple, list),
            )
            or len(self.image_shape) != 2
            or any(
                isinstance(value, (bool, np.bool_))
                or not isinstance(
                    value,
                    (int, np.integer),
                )
                or int(value) < 1
                for value in self.image_shape
            )
        ):
            raise ValueError(
                "image_shape must contain two positive dimensions."
            )

        shape = tuple(
            int(value)
            for value in self.image_shape
        )

        order = _positive_integer(
            self.max_neighbors,
            name="max_neighbors",
            maximum=255,
        )
        minimum = _positive_fraction(
            self.min_conditional_variance,
            name="min_conditional_variance",
        )

        valid = _readonly(
            self.valid_flat_indices,
            np.int64,
        )
        indptr = _readonly(
            self.neighbor_indptr,
            np.int64,
        )
        neighbors = _readonly(
            self.neighbor_indices,
            np.int32,
        )
        weights = _readonly(
            self.weights,
            np.float64,
        )
        variances = _readonly(
            self.conditional_variances,
            np.float64,
        )

        n_valid = valid.size

        if valid.ndim != 1 or n_valid < 1:
            raise ValueError(
                "valid_flat_indices must be a non-empty vector."
            )

        if (
            np.any(valid < 0)
            or np.any(valid >= np.prod(shape))
        ):
            raise ValueError(
                "valid_flat_indices lie outside image_shape."
            )

        if np.any(np.diff(valid) <= 0):
            raise ValueError(
                "valid_flat_indices must be strictly increasing "
                "row-major indices."
            )

        if indptr.shape != (n_valid + 1,):
            raise ValueError(
                "neighbor_indptr must have n_valid + 1 entries."
            )

        if (
            indptr[0] != 0
            or np.any(np.diff(indptr) < 0)
        ):
            raise ValueError(
                "neighbor_indptr must start at zero and be monotonic."
            )

        if (
            neighbors.ndim != 1
            or weights.ndim != 1
        ):
            raise ValueError(
                "neighbor_indices and weights must be vectors."
            )

        if (
            indptr[-1] != neighbors.size
            or weights.size != neighbors.size
        ):
            raise ValueError(
                "Vecchia CSR array lengths are inconsistent."
            )

        counts = np.diff(indptr)
        expected_counts = np.minimum(
            np.arange(
                n_valid,
                dtype=np.int64,
            ),
            order,
        )

        if not np.array_equal(
            counts,
            expected_counts,
        ):
            raise ValueError(
                "Each Vecchia row must contain "
                "min(row, max_neighbors) links."
            )

        if not np.all(np.isfinite(weights)):
            raise ValueError(
                "Vecchia weights must be finite."
            )

        if variances.shape != (n_valid,):
            raise ValueError(
                "conditional_variances must have n_valid entries."
            )

        if (
            not np.all(np.isfinite(variances))
            or np.any(variances <= minimum)
        ):
            raise ValueError(
                "Vecchia conditional variances must exceed "
                "the recorded minimum."
            )

        if np.any(
            variances > 1.0 + 1.0e-10
        ):
            raise ValueError(
                "Vecchia conditional variances cannot exceed one."
            )

        _validate_csr_predecessors(
            indptr,
            neighbors,
            n_valid,
        )

        logdet = float(
            self.correlation_logdet
        )
        expected_logdet = float(
            np.sum(
                np.log(variances),
                dtype=np.float64,
            )
        )

        if (
            not np.isfinite(logdet)
            or not np.isclose(
                logdet,
                expected_logdet,
                rtol=1.0e-12,
                atol=1.0e-12,
            )
        ):
            raise ValueError(
                "correlation_logdet must equal the sum "
                "of log conditional variances."
            )

        for name in (
            "correlation_kernel_sha256",
            "valid_mask_sha256",
        ):
            digest = getattr(self, name)

            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(
                    character
                    not in "0123456789abcdef"
                    for character in digest
                )
            ):
                raise ValueError(
                    f"{name} must be a lowercase SHA-256 digest."
                )

        if (
            isinstance(
                self.n_unique_patterns,
                (bool, np.bool_),
            )
            or not isinstance(
                self.n_unique_patterns,
                (int, np.integer),
            )
        ):
            raise TypeError(
                "n_unique_patterns must be an integer."
            )

        patterns = int(
            self.n_unique_patterns
        )

        if patterns < 1 or patterns > n_valid:
            raise ValueError(
                "n_unique_patterns must lie between 1 and n_valid."
            )

        if (
            self.schema
            != VECCHIA_FACTORIZATION_SCHEMA
        ):
            raise ValueError(
                "Unsupported Vecchia factorization schema."
            )

        if (
            self.version
            != VECCHIA_FACTORIZATION_VERSION
        ):
            raise ValueError(
                "Unsupported Vecchia factorization version."
            )

        if (
            self.approximation
            != VECCHIA_ALGORITHM
        ):
            raise ValueError(
                "Unsupported Vecchia approximation identifier."
            )

        if self.ordering != VECCHIA_ORDERING:
            raise ValueError(
                "Vecchia factorization ordering must be row_major."
            )

        object.__setattr__(
            self,
            "image_shape",
            shape,
        )
        object.__setattr__(
            self,
            "valid_flat_indices",
            valid,
        )
        object.__setattr__(
            self,
            "neighbor_indptr",
            indptr,
        )
        object.__setattr__(
            self,
            "neighbor_indices",
            neighbors,
        )
        object.__setattr__(
            self,
            "weights",
            weights,
        )
        object.__setattr__(
            self,
            "conditional_variances",
            variances,
        )
        object.__setattr__(
            self,
            "max_neighbors",
            order,
        )
        object.__setattr__(
            self,
            "correlation_logdet",
            logdet,
        )
        object.__setattr__(
            self,
            "n_unique_patterns",
            patterns,
        )
        object.__setattr__(
            self,
            "min_conditional_variance",
            minimum,
        )

    @property
    def n_valid_pixels(self) -> int:
        """Return the number of evaluated pixels."""
        return int(
            self.valid_flat_indices.size
        )

    @property
    def n_neighbor_links(self) -> int:
        """Return the number of conditional links."""
        return int(
            self.neighbor_indices.size
        )

    @property
    def storage_nbytes(self) -> int:
        """Return the factorization array footprint in bytes."""
        arrays = (
            self.valid_flat_indices,
            self.neighbor_indptr,
            self.neighbor_indices,
            self.weights,
            self.conditional_variances,
        )
        return int(
            sum(
                array.nbytes
                for array in arrays
            )
        )

    def validate_domain(
        self,
        *,
        mask: np.ndarray,
        correlation_kernel: np.ndarray,
        max_neighbors: int | None = None,
        min_conditional_variance: float | None = None,
    ) -> None:
        """Verify that the factorization matches its exact statistical domain."""
        mask_values = _mask_array(mask)
        kernel = _kernel_array(
            correlation_kernel
        )
        expected_valid = np.flatnonzero(
            ~mask_values.ravel(order="C")
        )

        if (
            mask_values.shape != self.image_shape
            or not np.array_equal(
                expected_valid,
                self.valid_flat_indices,
            )
        ):
            raise ValueError(
                "Vecchia factorization does not match "
                "the fitting mask."
            )

        if (
            _array_sha256(mask_values)
            != self.valid_mask_sha256
        ):
            raise ValueError(
                "Vecchia fitting-mask fingerprint does not match."
            )

        if (
            _array_sha256(kernel)
            != self.correlation_kernel_sha256
        ):
            raise ValueError(
                "Vecchia correlation-kernel fingerprint does not match."
            )

        if max_neighbors is not None:
            expected_neighbors = _positive_integer(
                max_neighbors,
                name="max_neighbors",
                maximum=255,
            )

            if (
                expected_neighbors
                != self.max_neighbors
            ):
                raise ValueError(
                    "Vecchia max_neighbors does not match "
                    "the factorization."
                )

        if (
            min_conditional_variance
            is not None
        ):
            expected_minimum = _positive_fraction(
                min_conditional_variance,
                name="min_conditional_variance",
            )

            if (
                expected_minimum
                != self.min_conditional_variance
            ):
                raise ValueError(
                    "Vecchia min_conditional_variance "
                    "does not match the factorization."
                )

    def metadata(
        self,
    ) -> dict[str, Any]:
        """Return explicit approximation and provenance metadata."""
        return {
            "schema": self.schema,
            "version": self.version,
            "approximation": self.approximation,
            "approximate": True,
            "ordering": self.ordering,
            "neighbor_policy": (
                VECCHIA_NEIGHBOR_POLICY
            ),
            "distance_metric": (
                VECCHIA_DISTANCE_METRIC
            ),
            "tie_break": VECCHIA_TIE_BREAK,
            "max_neighbors": self.max_neighbors,
            "min_conditional_variance": (
                self.min_conditional_variance
            ),
            "n_valid_pixels": (
                self.n_valid_pixels
            ),
            "n_neighbor_links": (
                self.n_neighbor_links
            ),
            "n_unique_patterns": (
                self.n_unique_patterns
            ),
            "storage_nbytes": (
                self.storage_nbytes
            ),
            "conditional_variance_min": float(
                np.min(
                    self.conditional_variances
                )
            ),
            "conditional_variance_max": float(
                np.max(
                    self.conditional_variances
                )
            ),
            "correlation_logdet": (
                self.correlation_logdet
            ),
            "correlation_kernel_sha256": (
                self.correlation_kernel_sha256
            ),
            "valid_mask_sha256": (
                self.valid_mask_sha256
            ),
        }


def _validate_csr_predecessors(
    indptr: np.ndarray,
    neighbors: np.ndarray,
    n_valid: int,
) -> None:
    counts = np.diff(indptr)
    chunk_size = 65536

    for start in range(
        0,
        n_valid,
        chunk_size,
    ):
        stop = min(
            start + chunk_size,
            n_valid,
        )
        values = neighbors[
            int(indptr[start]) :
            int(indptr[stop])
        ]
        rows = np.repeat(
            np.arange(
                start,
                stop,
                dtype=np.int32,
            ),
            counts[start:stop],
        )

        if (
            np.any(values < 0)
            or np.any(values >= rows)
        ):
            raise ValueError(
                "Every Vecchia neighbor must precede "
                "its target pixel."
            )

        if values.size > 1:
            order = np.lexsort(
                (values, rows)
            )
            sorted_rows = rows[order]
            sorted_values = values[order]
            duplicate = (
                (
                    sorted_rows[1:]
                    == sorted_rows[:-1]
                )
                & (
                    sorted_values[1:]
                    == sorted_values[:-1]
                )
            )

            if np.any(duplicate):
                raise ValueError(
                    "Vecchia neighbor rows cannot "
                    "contain duplicates."
                )


def _ordered_previous(
    coordinates: np.ndarray,
    row: int,
    candidates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    previous = np.asarray(
        candidates[candidates < row],
        dtype=np.int64,
    )
    differences = (
        coordinates[previous]
        - coordinates[row]
    )
    distance_squared = np.sum(
        differences * differences,
        axis=1,
    )
    order = np.lexsort(
        (
            -previous,
            distance_squared,
        )
    )
    return (
        previous[order],
        distance_squared[order],
    )


def _fallback_neighbors(
    tree: cKDTree,
    coordinates: np.ndarray,
    row: int,
    wanted: int,
) -> np.ndarray:
    if np.all(
        coordinates[:row, 0]
        == coordinates[row, 0]
    ):
        ordered, _ = _ordered_previous(
            coordinates,
            row,
            np.arange(
                row,
                dtype=np.int64,
            ),
        )
        return ordered[:wanted].astype(
            np.int32,
            copy=False,
        )

    query_size = min(
        coordinates.shape[0],
        max(
            4 * wanted + 1,
            65,
        ),
    )

    while True:
        _, queried = tree.query(
            coordinates[row],
            k=query_size,
            workers=1,
        )
        queried = np.atleast_1d(
            queried
        ).astype(
            np.int64,
            copy=False,
        )

        ordered, distances = (
            _ordered_previous(
                coordinates,
                row,
                queried,
            )
        )

        if ordered.size >= wanted:
            queried_delta = (
                coordinates[queried]
                - coordinates[row]
            )
            query_limit = int(
                np.max(
                    np.sum(
                        queried_delta
                        * queried_delta,
                        axis=1,
                    )
                )
            )
            complete = (
                query_size
                == coordinates.shape[0]
                or (
                    distances[wanted - 1]
                    < query_limit
                )
            )

            if complete:
                return ordered[
                    :wanted
                ].astype(
                    np.int32,
                    copy=False,
                )

        if (
            query_size
            == coordinates.shape[0]
        ):
            raise RuntimeError(
                "Global predecessor search did not converge."
            )

        query_size = min(
            coordinates.shape[0],
            2 * query_size,
        )


def _select_global_neighbors(
    mask: np.ndarray,
    max_neighbors: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Select global nearest predecessors with a reproducible distance tie."""
    valid_flat = np.flatnonzero(
        ~mask.ravel(order="C")
    )
    coordinates = np.column_stack(
        np.unravel_index(
            valid_flat,
            mask.shape,
        )
    )
    n_valid = valid_flat.size

    selected = np.full(
        (
            n_valid,
            max_neighbors,
        ),
        -1,
        dtype=np.int32,
    )

    tree = cKDTree(coordinates)
    query_size = min(
        n_valid,
        max(
            4 * max_neighbors + 1,
            65,
        ),
    )
    batch_size = 2048

    for start in range(
        0,
        n_valid,
        batch_size,
    ):
        stop = min(
            start + batch_size,
            n_valid,
        )
        rows = np.arange(
            start,
            stop,
            dtype=np.int64,
        )

        _, queried = tree.query(
            coordinates[rows],
            k=query_size,
            workers=1,
        )

        if query_size == 1:
            queried = queried[:, None]

        queried = np.asarray(
            queried,
            dtype=np.int64,
        )
        differences = (
            coordinates[queried]
            - coordinates[rows, None, :]
        )
        distance_squared = np.sum(
            differences * differences,
            axis=2,
        )
        eligible = (
            queried
            < rows[:, None]
        )
        distance_key = np.where(
            eligible,
            distance_squared,
            np.iinfo(
                np.int64
            ).max,
        )
        predecessor_key = np.where(
            eligible,
            -queried,
            np.iinfo(
                np.int64
            ).max,
        )
        order = np.lexsort(
            (
                predecessor_key,
                distance_key,
            ),
            axis=1,
        )
        ordered = np.take_along_axis(
            queried,
            order,
            axis=1,
        )
        ordered_distances = (
            np.take_along_axis(
                distance_key,
                order,
                axis=1,
            )
        )
        available = np.sum(
            eligible,
            axis=1,
        )
        wanted = np.minimum(
            rows,
            max_neighbors,
        )
        query_limit = np.max(
            distance_squared,
            axis=1,
        )

        for local, row in enumerate(
            rows
        ):
            count = int(
                wanted[local]
            )

            if count == 0:
                continue

            complete = (
                available[local] >= count
                and (
                    query_size == n_valid
                    or (
                        ordered_distances[
                            local,
                            count - 1,
                        ]
                        < query_limit[local]
                    )
                )
            )

            if complete:
                selected[
                    row,
                    :count,
                ] = ordered[
                    local,
                    :count,
                ]
            else:
                selected[
                    row,
                    :count,
                ] = _fallback_neighbors(
                    tree,
                    coordinates,
                    int(row),
                    count,
                )

    return coordinates, selected


def _kernel_at_offsets(
    kernel: np.ndarray,
    dy: np.ndarray,
    dx: np.ndarray,
) -> np.ndarray:
    dy_values, dx_values = (
        np.broadcast_arrays(
            dy,
            dx,
        )
    )
    output = np.zeros(
        dy_values.shape,
        dtype=np.float64,
    )
    half_y = (
        kernel.shape[0] // 2
    )
    half_x = (
        kernel.shape[1] // 2
    )
    inside = (
        (
            np.abs(dy_values)
            <= half_y
        )
        & (
            np.abs(dx_values)
            <= half_x
        )
    )

    output[inside] = kernel[
        dy_values[inside] + half_y,
        dx_values[inside] + half_x,
    ]
    return output


def _conditional_coefficients(
    offsets: tuple[
        tuple[int, int],
        ...,
    ],
    kernel: np.ndarray,
    minimum: float,
) -> tuple[np.ndarray, float]:
    if not offsets:
        return (
            np.empty(
                0,
                dtype=np.float64,
            ),
            1.0,
        )

    coordinates = np.asarray(
        offsets,
        dtype=np.int64,
    )
    covariance_to_target = (
        _kernel_at_offsets(
            kernel,
            -coordinates[:, 0],
            -coordinates[:, 1],
        )
    )
    differences = (
        coordinates[:, None, :]
        - coordinates[None, :, :]
    )
    covariance_between_neighbors = (
        _kernel_at_offsets(
            kernel,
            differences[..., 0],
            differences[..., 1],
        )
    )

    try:
        factor = cho_factor(
            covariance_between_neighbors,
            lower=True,
            check_finite=False,
        )
        weights = cho_solve(
            factor,
            covariance_to_target,
            check_finite=False,
        )
    except np.linalg.LinAlgError as error:
        raise ValueError(
            "Vecchia neighbor covariance is not "
            "positive definite."
        ) from error

    variance = float(
        1.0
        - np.dot(
            covariance_to_target,
            weights,
        )
    )

    if (
        not np.all(np.isfinite(weights))
        or not np.isfinite(variance)
    ):
        raise ValueError(
            "Vecchia conditional coefficients are not finite."
        )

    if variance <= minimum:
        raise ValueError(
            "Vecchia conditional variance is not safely positive: "
            f"{variance:.6e} <= {minimum:.6e}."
        )

    if variance > 1.0 + 1.0e-10:
        raise ValueError(
            "Vecchia conditional variance exceeds one."
        )

    return (
        np.asarray(
            weights,
            dtype=np.float64,
        ),
        variance,
    )


def build_vecchia_factorization(
    mask: np.ndarray,
    correlation_kernel: np.ndarray,
    *,
    max_neighbors: int = 64,
    min_conditional_variance: float = 1.0e-12,
) -> VecchiaFactorization:
    """Build a proper row-major global-neighbour Gaussian factorization."""
    mask_values = _mask_array(mask)
    kernel = _kernel_array(
        correlation_kernel
    )
    max_neighbors = _positive_integer(
        max_neighbors,
        name="max_neighbors",
        maximum=255,
    )
    minimum = _positive_fraction(
        min_conditional_variance,
        name="min_conditional_variance",
    )
    valid_flat = np.flatnonzero(
        ~mask_values.ravel(order="C")
    )

    if (
        valid_flat.size
        > np.iinfo(np.int32).max
    ):
        raise ValueError(
            "Vecchia valid-pixel count exceeds "
            "int32 index capacity."
        )

    coordinates, selected = (
        _select_global_neighbors(
            mask_values,
            max_neighbors,
        )
    )

    n_valid = valid_flat.size
    maximum_links = (
        n_valid * max_neighbors
    )

    neighbor_indices = np.empty(
        maximum_links,
        dtype=np.int32,
    )
    weights = np.empty(
        maximum_links,
        dtype=np.float64,
    )
    indptr = np.zeros(
        n_valid + 1,
        dtype=np.int64,
    )
    variances = np.ones(
        n_valid,
        dtype=np.float64,
    )

    cache: OrderedDict[
        tuple[
            tuple[int, int],
            ...,
        ],
        tuple[
            np.ndarray,
            float,
        ],
    ] = OrderedDict()

    pattern_digests: set[bytes] = set()
    cursor = 0

    for row in range(n_valid):
        count = min(
            row,
            max_neighbors,
        )
        row_neighbors = selected[
            row,
            :count,
        ]
        relative = (
            coordinates[row_neighbors]
            - coordinates[row]
        )
        key = tuple(
            (
                int(dy),
                int(dx),
            )
            for dy, dx in relative
        )

        pattern_digests.add(
            hashlib.sha256(
                np.ascontiguousarray(
                    relative,
                    dtype=np.int64,
                ).tobytes()
            ).digest()
        )

        coefficients = cache.pop(
            key,
            None,
        )

        if coefficients is None:
            coefficients = (
                _conditional_coefficients(
                    key,
                    kernel,
                    minimum,
                )
            )

        cache[key] = coefficients

        if (
            len(cache)
            > _COEFFICIENT_CACHE_SIZE
        ):
            cache.popitem(
                last=False
            )

        (
            row_weights,
            variance,
        ) = coefficients

        neighbor_indices[
            cursor :
            cursor + count
        ] = row_neighbors
        weights[
            cursor :
            cursor + count
        ] = row_weights

        cursor += count
        indptr[row + 1] = cursor
        variances[row] = variance

    neighbor_indices.resize(
        cursor,
        refcheck=False,
    )
    weights.resize(
        cursor,
        refcheck=False,
    )

    logdet = float(
        np.sum(
            np.log(variances),
            dtype=np.float64,
        )
    )

    return VecchiaFactorization(
        image_shape=mask_values.shape,
        valid_flat_indices=valid_flat,
        neighbor_indptr=indptr,
        neighbor_indices=neighbor_indices,
        weights=weights,
        conditional_variances=variances,
        max_neighbors=max_neighbors,
        correlation_logdet=logdet,
        correlation_kernel_sha256=(
            _array_sha256(kernel)
        ),
        valid_mask_sha256=(
            _array_sha256(mask_values)
        ),
        n_unique_patterns=(
            len(pattern_digests)
        ),
        min_conditional_variance=minimum,
    )


class VecchiaCorrelatedGaussianImageLikelihood(
    ImageLikelihood
):
    """Normalized ``SRS`` Gaussian likelihood under Vecchia-``m``.

    ``S`` is the fixed marginal scale ``noise``, or, with an
    ``effective_gain`` ``g``, the scale of AD-047,
    ``sqrt(noise**2 + max(m, 0) / g)``, computed at every evaluation from the
    source part ``m`` of the model; ``R`` is factorized once.
    """

    def __init__(
        self,
        data: np.ndarray,
        noise: np.ndarray | float,
        correlation_kernel: np.ndarray,
        mask: np.ndarray | None = None,
        *,
        factorization: (
            VecchiaFactorization
            | None
        ) = None,
        max_neighbors: int = 64,
        min_conditional_variance: float = 1.0e-12,
        include_normalization: bool = True,
        effective_gain: float | None = None,
    ) -> None:
        super().__init__(
            data=data,
            mask=mask,
        )

        if len(self.image_shape) != 2:
            raise ValueError(
                "Vecchia likelihood requires two-dimensional data."
            )

        if effective_gain is not None:
            if isinstance(effective_gain, bool) or not (
                np.isfinite(float(effective_gain))
                and float(effective_gain) > 0.0
            ):
                raise ValueError(
                    "effective_gain must be a positive finite number."
                )

            if not include_normalization:
                raise ValueError(
                    "The source variance of AD-047 makes the log determinant "
                    "of the covariance depend on the parameters, so its "
                    "normalization cannot be disabled."
                )

        self.noise_values = (
            self.prepare_fixed_array(
                noise,
                name="Noise",
                require_positive=True,
            )
        )
        kernel = _kernel_array(
            correlation_kernel
        )
        domain_mask = (
            np.zeros(
                self.image_shape,
                dtype=bool,
            )
            if self.mask is None
            else np.asarray(
                self.mask,
                dtype=bool,
            )
        )
        order = _positive_integer(
            max_neighbors,
            name="max_neighbors",
            maximum=255,
        )
        minimum = _positive_fraction(
            min_conditional_variance,
            name="min_conditional_variance",
        )

        if factorization is None:
            factorization = (
                build_vecchia_factorization(
                    domain_mask,
                    kernel,
                    max_neighbors=order,
                    min_conditional_variance=(
                        minimum
                    ),
                )
            )
        elif not isinstance(
            factorization,
            VecchiaFactorization,
        ):
            raise TypeError(
                "factorization must be a VecchiaFactorization."
            )

        factorization.validate_domain(
            mask=domain_mask,
            correlation_kernel=kernel,
            max_neighbors=order,
            min_conditional_variance=minimum,
        )

        self.correlation_kernel = kernel
        self.factorization = factorization
        self.correlation_logdet = (
            factorization.correlation_logdet
        )
        self.include_normalization = bool(
            include_normalization
        )
        self._weight_operator = csr_matrix(
            (
                factorization.weights,
                factorization.neighbor_indices,
                factorization.neighbor_indptr,
            ),
            shape=(
                self.n_valid_pixels,
                self.n_valid_pixels,
            ),
            copy=False,
        )

        self.effective_gain = (
            None
            if effective_gain is None
            else float(effective_gain)
        )
        self.requires_source = self.effective_gain is not None
        self._sky_variance = np.square(
            self.noise_values
        )
        self._normalization_constant = -0.5 * (
            (
                self.n_valid_pixels
                * np.log(
                    2.0 * np.pi
                )
            )
            + factorization.correlation_logdet
        )

        if self.include_normalization:
            self._normalization = -0.5 * (
                (
                    self.n_valid_pixels
                    * np.log(
                        2.0 * np.pi
                    )
                )
                + (
                    2.0
                    * np.sum(
                        np.log(
                            self.noise_values
                        ),
                        dtype=np.float64,
                    )
                )
                + factorization.correlation_logdet
            )
        else:
            self._normalization = 0.0

    @property
    def approximation_metadata(
        self,
    ) -> dict[str, Any]:
        """Return evidence-relevant approximation metadata."""
        metadata = (
            self.factorization.metadata()
        )
        metadata[
            "normalization_complete"
        ] = self.include_normalization
        metadata[
            "evidence_scope"
        ] = (
            "approximate_absolute"
            if self.include_normalization
            else "posterior_only"
        )
        metadata[
            "effective_gain"
        ] = self.effective_gain
        return metadata

    def marginal_scale(
        self,
        source: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return the marginal scale ``S`` in the evaluated pixels.

        Parameters
        ----------
        source : array-like, optional
            Source part of the model, without the residual background;
            required with an effective gain.
        """
        if self.effective_gain is None:
            return self.noise_values

        if source is None:
            raise ValueError(
                "The scale of AD-047 requires the source part of the model."
            )

        return np.sqrt(
            self._sky_variance
            + np.clip(
                self.prepare_model(source),
                0.0,
                None,
            )
            / self.effective_gain
        )

    def conditional_innovations(
        self,
        model: np.ndarray,
        source: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return unscaled conditional innovations."""
        model_values = (
            self.prepare_model(model)
        )
        standardized = (
            self.data_values
            - model_values
        ) / self.marginal_scale(source)
        output = self._innovations(
            standardized
        )
        output.setflags(write=False)
        return output

    def marginal_standardized_residual(
        self,
        model: np.ndarray,
        source: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return ``(data - model) / S`` on-grid, with NaN when masked."""
        standardized = (
            self.data_values
            - self.prepare_model(model)
        ) / self.marginal_scale(source)

        output = np.full(
            self.image_shape,
            np.nan,
            dtype=float,
        )
        output[
            self._valid_pixels
        ] = standardized
        return output

    def standardized_conditional_innovations(
        self,
        model: np.ndarray,
        source: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return conditional innovations divided by their standard deviations."""
        output = (
            self.conditional_innovations(
                model,
                source,
            ).copy()
        )
        output /= np.sqrt(
            self.factorization.conditional_variances
        )
        output.setflags(write=False)
        return output

    def quadratic_innovations(
        self,
        model: np.ndarray,
        source: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return per-pixel normalized conditional quadratic terms."""
        output = self._quadratic_terms(
            self.prepare_model(model),
            self.marginal_scale(source),
        )
        output.setflags(write=False)
        return output

    def quadratic_form(
        self,
        model: np.ndarray,
        source: np.ndarray | None = None,
    ) -> float:
        """Return the Vecchia conditional quadratic form."""
        return self._quadratic(
            self.prepare_model(model),
            self.marginal_scale(source),
        )

    def _innovations(
        self,
        standardized: np.ndarray,
    ) -> np.ndarray:
        return np.asarray(
            standardized
            - (
                self._weight_operator
                @ standardized
            ),
            dtype=np.float64,
        )

    def _quadratic_terms(
        self,
        model_values: np.ndarray,
        scale: np.ndarray,
    ) -> np.ndarray:
        standardized = (
            self.data_values
            - model_values
        ) / scale
        innovations = self._innovations(
            standardized
        )
        terms = np.asarray(
            np.square(
                innovations
            )
            / self.factorization.conditional_variances,
            dtype=np.float64,
        )

        if (
            not np.all(np.isfinite(terms))
            or np.any(terms < 0.0)
        ):
            raise RuntimeError(
                "Vecchia quadratic terms must be "
                "finite and non-negative."
            )

        return terms

    def _quadratic(
        self,
        model_values: np.ndarray,
        scale: np.ndarray,
    ) -> float:
        quadratic = float(
            np.sum(
                self._quadratic_terms(
                    model_values,
                    scale,
                ),
                dtype=np.float64,
            )
        )

        if (
            not np.isfinite(quadratic)
            or quadratic < 0.0
        ):
            raise RuntimeError(
                "Vecchia quadratic form is not "
                "finite and non-negative."
            )

        return quadratic

    def _log_likelihood(
        self,
        model_values: np.ndarray,
    ) -> float:
        return float(
            -0.5
            * self._quadratic(
                model_values,
                self.noise_values,
            )
            + self._normalization
        )

    def _log_likelihood_with_source(
        self,
        model_values: np.ndarray,
        source_values: np.ndarray,
    ) -> float:
        scale = np.sqrt(
            self._sky_variance
            + np.clip(
                source_values,
                0.0,
                None,
            )
            / self.effective_gain
        )

        return float(
            -0.5
            * self._quadratic(
                model_values,
                scale,
            )
            + self._normalization_constant
            - np.sum(
                np.log(
                    scale
                ),
                dtype=np.float64,
            )
        )
