"""Interactive mask-action editor."""

import numpy as np

from baymorph.pipeline import refine_mask


class MaskActionEditor:
    """Interactive editor for reproducible mask-refinement actions."""

    def __init__(
        self,
        image,
        initial_mask,
        target_x=None,
        target_y=None,
        display_floor=1e-3,
    ):
        self.image = np.asarray(
            image,
            dtype=float,
        )

        self.initial_mask = np.asarray(
            initial_mask,
            dtype=bool,
        )

        self.mask = self.initial_mask.copy()

        self.actions = []

        self.target_x = target_x
        self.target_y = target_y
        self.display_floor = display_floor

        self._built = False

    def _require_widgets(
        self,
    ):
        try:
            import ipywidgets as widgets
            import plotly.graph_objects as go
            from IPython.display import display

        except ImportError as exc:
            raise ImportError(
                "MaskActionEditor requires plotly, ipywidgets, "
                "and IPython. Install them with: "
                "pip install plotly ipywidgets anywidget"
            ) from exc

        return widgets, go, display

    def _build_controls(
        self,
    ):
        widgets, go, display = self._require_widgets()

        self.mode_widget = widgets.Dropdown(
            options=[
                "source",
                "star",
            ],
            value="source",
            description="Mode",
        )

        self.threshold_widget = widgets.FloatSlider(
            value=1.5,
            min=0.5,
            max=5.0,
            step=0.1,
            description="sigma",
        )

        self.dilation_widget = widgets.IntSlider(
            value=5,
            min=0,
            max=25,
            step=1,
            description="dilation",
        )

        self.box_widget = widgets.IntSlider(
            value=81,
            min=31,
            max=201,
            step=10,
            description="box",
        )

        self.radius_widget = widgets.IntSlider(
            value=70,
            min=10,
            max=200,
            step=5,
            description="radius",
        )

        self.undo_button = widgets.Button(
            description="Undo",
        )

        self.reset_button = widgets.Button(
            description="Reset",
        )

        self.output = widgets.Output()

        self._widgets = widgets
        self._go = go
        self._display = display

    def _build_figure(
        self,
    ):
        z = np.log10(
            np.maximum(
                self.image,
                self.display_floor,
            )
        )

        self.fig = self._go.FigureWidget()

        self.fig.add_trace(
            self._go.Heatmap(
                z=z,
                colorscale="Viridis",
                showscale=True,
                name="Image",
            )
        )

        self.fig.add_trace(
            self._go.Contour(
                z=self.initial_mask.astype(float),
                contours=dict(
                    start=0.5,
                    end=0.5,
                    size=1,
                    coloring="none",
                ),
                line=dict(
                    color="white",
                    width=1,
                ),
                showscale=False,
                name="Initial mask",
            )
        )

        self.fig.add_trace(
            self._go.Contour(
                z=self.mask.astype(float),
                contours=dict(
                    start=0.5,
                    end=0.5,
                    size=1,
                    coloring="none",
                ),
                line=dict(
                    color="red",
                    width=2,
                ),
                showscale=False,
                name="Refined mask",
            )
        )

        if (
            self.target_x is not None
            and self.target_y is not None
        ):
            self.fig.add_trace(
                self._go.Scatter(
                    x=[
                        self.target_x,
                    ],
                    y=[
                        self.target_y,
                    ],
                    mode="markers",
                    marker=dict(
                        symbol="x",
                        size=12,
                        color="cyan",
                    ),
                    name="Target",
                )
            )

        self.fig.update_layout(
            width=750,
            height=750,
            title="Click contaminants to refine mask",
            yaxis=dict(
                scaleanchor="x",
                autorange="reversed",
            ),
        )

    def _connect_callbacks(
        self,
    ):
        for trace in self.fig.data:
            trace.on_click(
                self._on_click,
            )

        self.undo_button.on_click(
            self._undo,
        )

        self.reset_button.on_click(
            self._reset,
        )

    def _build(
        self,
    ):
        self._build_controls()
        self._build_figure()
        self._connect_callbacks()
        self._built = True

    def _build_action(
        self,
        x,
        y,
    ):
        action = {
            "type": self.mode_widget.value,
            "x": float(x),
            "y": float(y),
            "threshold_sigma": float(
                self.threshold_widget.value,
            ),
            "dilation_iterations": int(
                self.dilation_widget.value,
            ),
        }

        if self.mode_widget.value == "source":
            action["box_size"] = int(
                self.box_widget.value,
            )

        if self.mode_widget.value == "star":
            action["radius"] = int(
                self.radius_widget.value,
            )

        return action

    def _recompute_mask(
        self,
    ):
        self.mask = refine_mask(
            image=self.image,
            mask=self.initial_mask,
            actions=self.actions,
        )

        self.fig.data[2].z = self.mask.astype(float)

    def _on_click(
        self,
        trace,
        points,
        state,
    ):
        if len(points.xs) == 0:
            return

        x = points.xs[0]
        y = points.ys[0]

        action = self._build_action(
            x,
            y,
        )

        self.actions.append(
            action,
        )

        self._recompute_mask()

        with self.output:
            print(action)

    def _undo(
        self,
        button,
    ):
        if len(self.actions) == 0:
            return

        removed = self.actions.pop()

        self._recompute_mask()

        with self.output:
            print(
                "Removed:",
                removed,
            )

    def _reset(
        self,
        button,
    ):
        self.actions = []

        self._recompute_mask()

        with self.output:
            print(
                "Reset mask actions.",
            )

    def show(
        self,
    ):
        """Display the interactive editor."""
        if not self._built:
            self._build()

        self._display(
            self._widgets.VBox(
                [
                    self._widgets.HBox(
                        [
                            self.mode_widget,
                            self.threshold_widget,
                            self.dilation_widget,
                        ]
                    ),
                    self._widgets.HBox(
                        [
                            self.box_widget,
                            self.radius_widget,
                            self.undo_button,
                            self.reset_button,
                        ]
                    ),
                    self.fig,
                    self.output,
                ]
            )
        )