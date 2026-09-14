"""Interactive target-centre selection for notebook workflows."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray


TargetPosition = tuple[float, float]


class TargetSelector:
    """Select and explicitly confirm a target position in a 2D image.

    The canonical coordinates are zero-based BayMorph pixel coordinates:
    ``x`` is the NumPy column index and ``y`` is the NumPy row index.
    Candidate positions remain unconfirmed until :meth:`confirm` is called.

    The interactive frontend is constructed lazily, so importing and testing
    BayMorph does not require notebook-only dependencies.
    """

    def __init__(
        self,
        image: NDArray[np.floating[Any]] | np.ndarray,
        *,
        initial_position: TargetPosition | None = None,
        title: str = "Select the target center",
        max_preview_size: int = 1000,
        on_confirm: Callable[[TargetPosition], None] | None = None,
        on_invalidate: Callable[[], None] | None = None,
    ) -> None:
        data = np.asarray(
            image,
            dtype=float,
        )

        if data.ndim != 2:
            raise ValueError("TargetSelector requires a two-dimensional image.")

        if data.size == 0:
            raise ValueError("TargetSelector requires a non-empty image.")

        if not np.any(np.isfinite(data)):
            raise ValueError("TargetSelector requires at least one finite pixel.")

        if max_preview_size <= 0:
            raise ValueError("max_preview_size must be positive.")

        self.image = data
        self.title = str(title)
        self.max_preview_size = int(max_preview_size)
        self._on_confirm = on_confirm
        self._on_invalidate = on_invalidate
        self._frontend: dict[str, Any] | None = None
        self._synchronizing = False

        height, width = self.image.shape

        if initial_position is None:
            initial_position = (
                0.5 * (width - 1),
                0.5 * (height - 1),
            )

        self._initial_position = self._validate_position(initial_position)
        self._candidate = self._initial_position
        self._value: TargetPosition | None = None

    @property
    def candidate(
        self,
    ) -> TargetPosition:
        """Return the current, possibly unconfirmed, candidate position."""
        return self._candidate

    @property
    def value(
        self,
    ) -> TargetPosition | None:
        """Return the confirmed position, or None before confirmation."""
        return self._value

    @property
    def is_confirmed(
        self,
    ) -> bool:
        """Return whether the current candidate was explicitly confirmed."""
        return self._value is not None

    def _validate_position(
        self,
        position: TargetPosition,
    ) -> TargetPosition:
        """Return a finite position inside the image bounds."""
        try:
            x, y = position
        except (
            TypeError,
            ValueError,
        ) as error:
            raise ValueError(
                "Target position must contain exactly two coordinates."
            ) from error

        x = float(x)
        y = float(y)

        if not np.all(
            np.isfinite(
                [
                    x,
                    y,
                ]
            )
        ):
            raise ValueError("Target coordinates must be finite.")

        height, width = self.image.shape

        if not (0.0 <= x <= width - 1 and 0.0 <= y <= height - 1):
            raise ValueError(
                "Target coordinates must lie inside the image: "
                f"0 <= x <= {width - 1} and "
                f"0 <= y <= {height - 1}."
            )

        return (
            x,
            y,
        )

    def set_candidate(
        self,
        x: float,
        y: float,
    ) -> TargetPosition:
        """Set an unconfirmed target candidate and update the frontend."""
        position = self._validate_position(
            (
                x,
                y,
            )
        )

        was_confirmed = self._value is not None

        self._candidate = position
        self._value = None

        if was_confirmed and self._on_invalidate is not None:
            self._on_invalidate()

        self._synchronize_frontend()

        return self._candidate

    def confirm(
        self,
    ) -> TargetPosition:
        """Confirm the current candidate and notify the owning advisor."""
        position = self._validate_position(self._candidate)

        if self._on_confirm is not None:
            self._on_confirm(position)

        self._value = position
        self._synchronize_frontend()

        return position

    def reset(
        self,
    ) -> TargetPosition:
        """Restore the initial candidate and clear any confirmation."""
        was_confirmed = self._value is not None

        self._candidate = self._initial_position
        self._value = None

        if was_confirmed and self._on_invalidate is not None:
            self._on_invalidate()

        self._synchronize_frontend()

        return self._candidate

    def _preview(
        self,
    ) -> tuple[
        NDArray[np.floating[Any]],
        NDArray[np.floating[Any]],
        NDArray[np.integer[Any]],
        NDArray[np.integer[Any]],
    ]:
        """Return a robust display stretch and its original coordinates."""
        height, width = self.image.shape
        stride = max(
            1,
            int(
                np.ceil(
                    max(
                        height,
                        width,
                    )
                    / self.max_preview_size
                )
            ),
        )

        raw = self.image[
            ::stride,
            ::stride,
        ]
        finite = np.isfinite(raw)
        values = raw[finite]

        lower, upper = np.percentile(
            values,
            [
                1.0,
                99.8,
            ],
        )

        scale = float(upper - lower)

        if not np.isfinite(scale) or scale <= 0.0:
            scale = max(
                float(np.nanmax(np.abs(values))),
                1.0,
            )

        filled = np.where(
            finite,
            raw,
            lower,
        )
        clipped = np.clip(
            filled,
            lower,
            upper,
        )
        display = np.arcsinh(10.0 * (clipped - lower) / scale)

        x = np.arange(
            0,
            width,
            stride,
            dtype=int,
        )
        y = np.arange(
            0,
            height,
            stride,
            dtype=int,
        )

        return (
            display,
            raw,
            x,
            y,
        )

    def _build_frontend(
        self,
    ) -> dict[str, Any]:
        """Construct the lazy Plotly and ipywidgets frontend."""
        try:
            import ipywidgets as widgets
            import plotly.graph_objects as go
        except ImportError as error:
            raise ImportError(
                "Interactive target selection requires the BayMorph "
                "notebook dependencies. Install them with "
                "`pip install 'baymorph[notebooks]'`."
            ) from error

        display, raw, x_values, y_values = self._preview()

        try:
            figure = go.FigureWidget(
                data=[
                    go.Heatmap(
                        z=display,
                        x=x_values,
                        y=y_values,
                        customdata=raw,
                        colorscale="Magma",
                        showscale=False,
                        hovertemplate=(
                            "X=%{x:.2f}<br>"
                            "Y=%{y:.2f}<br>"
                            "Value=%{customdata:.6g}"
                            "<extra></extra>"
                        ),
                    ),
                    go.Scatter(
                        x=[self._candidate[0]],
                        y=[self._candidate[1]],
                        mode="markers",
                        marker={
                            "symbol": "x",
                            "size": 18,
                            "color": "cyan",
                            "line": {
                                "color": "black",
                                "width": 2,
                            },
                        },
                        hoverinfo="skip",
                        showlegend=False,
                    ),
                ]
            )
        except Exception as error:
            raise RuntimeError(
                "Plotly FigureWidget could not be initialized. Install "
                "the complete notebook environment with "
                "`pip install 'baymorph[notebooks]'`."
            ) from error

        height, width = self.image.shape

        figure.update_layout(
            title={
                "text": self.title,
                "x": 0.5,
            },
            template="plotly_white",
            height=700,
            margin={
                "l": 70,
                "r": 20,
                "t": 70,
                "b": 70,
            },
            dragmode="pan",
            xaxis={
                "title": "X [pixel]",
                "range": [
                    -0.5,
                    width - 0.5,
                ],
                "constrain": "domain",
            },
            yaxis={
                "title": "Y [pixel]",
                "range": [
                    -0.5,
                    height - 0.5,
                ],
                "scaleanchor": "x",
                "scaleratio": 1,
                "constrain": "domain",
            },
        )

        x_field = widgets.BoundedFloatText(
            value=self._candidate[0],
            min=0.0,
            max=float(width - 1),
            step=0.1,
            description="X:",
        )
        y_field = widgets.BoundedFloatText(
            value=self._candidate[1],
            min=0.0,
            max=float(height - 1),
            step=0.1,
            description="Y:",
        )

        confirm_button = widgets.Button(
            description="Confirm center",
            button_style="success",
            icon="check",
        )
        reset_button = widgets.Button(
            description="Reset",
            icon="undo",
        )
        status = widgets.HTML()

        instructions = widgets.HTML(
            "<b>Select the galaxy center.</b> "
            "Click the image or enter X and Y manually, then press "
            "<i>Confirm center</i>.<br>"
            "BayMorph coordinates are zero-based: "
            "<code>image[y, x]</code>. Use the Plotly toolbar to zoom or pan."
        )

        controls = widgets.HBox(
            [
                x_field,
                y_field,
                confirm_button,
                reset_button,
            ]
        )
        container = widgets.VBox(
            [
                instructions,
                figure,
                controls,
                status,
            ]
        )

        frontend = {
            "container": container,
            "figure": figure,
            "image_trace": figure.data[0],
            "marker_trace": figure.data[1],
            "x_field": x_field,
            "y_field": y_field,
            "status": status,
        }

        def handle_click(
            _trace: Any,
            points: Any,
            _state: Any,
        ) -> None:
            if not points.xs or not points.ys:
                return

            self.set_candidate(
                points.xs[0],
                points.ys[0],
            )

        def handle_field_change(
            _change: Any,
        ) -> None:
            if self._synchronizing:
                return

            self.set_candidate(
                x_field.value,
                y_field.value,
            )

        def handle_confirm(
            _button: Any,
        ) -> None:
            self.confirm()

        def handle_reset(
            _button: Any,
        ) -> None:
            self.reset()

        frontend["image_trace"].on_click(handle_click)
        x_field.observe(
            handle_field_change,
            names="value",
        )
        y_field.observe(
            handle_field_change,
            names="value",
        )
        confirm_button.on_click(handle_confirm)
        reset_button.on_click(handle_reset)

        return frontend

    def _synchronize_frontend(
        self,
    ) -> None:
        """Synchronize fields, marker, crosshair, and status text."""
        if self._frontend is None:
            return

        self._synchronizing = True

        try:
            x, y = self._candidate
            frontend = self._frontend

            frontend["x_field"].value = x
            frontend["y_field"].value = y

            frontend["marker_trace"].x = [x]
            frontend["marker_trace"].y = [y]

            height, width = self.image.shape

            frontend["figure"].layout.shapes = (
                {
                    "type": "line",
                    "x0": x,
                    "x1": x,
                    "y0": -0.5,
                    "y1": height - 0.5,
                    "line": {
                        "color": "cyan",
                        "width": 1,
                        "dash": "dot",
                    },
                },
                {
                    "type": "line",
                    "x0": -0.5,
                    "x1": width - 0.5,
                    "y0": y,
                    "y1": y,
                    "line": {
                        "color": "cyan",
                        "width": 1,
                        "dash": "dot",
                    },
                },
            )

            if self.is_confirmed:
                frontend["status"].value = (
                    "<span style='color:green'><b>Confirmed:</b> "
                    f"X={x:.3f}, Y={y:.3f}</span>"
                )
            else:
                frontend["status"].value = (
                    "<span style='color:#b36b00'><b>Candidate:</b> "
                    f"X={x:.3f}, Y={y:.3f}. "
                    "Confirmation required.</span>"
                )
        finally:
            self._synchronizing = False

    @property
    def widget(
        self,
    ) -> Any:
        """Return the lazily constructed notebook widget."""
        if self._frontend is None:
            self._frontend = self._build_frontend()
            self._synchronize_frontend()

        return self._frontend["container"]

    def _ipython_display_(
        self,
    ) -> None:
        """Display the selector as the final expression of a notebook cell."""
        try:
            from IPython.display import display
        except ImportError as error:
            raise ImportError("Displaying TargetSelector requires IPython.") from error

        display(self.widget)


__all__ = [
    "TargetPosition",
    "TargetSelector",
]
