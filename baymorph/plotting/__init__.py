"""Plotting utilities for BayMorph."""

from .corner import (
    corner_plot,
    erf_sigma_to_percentiles,
)
from .images import (
    plot_fit_summary,
    plot_log_data_model_residual,
    plot_log_image,
    show_log_image,
)
from .posterior import (
    plot_1d_posterior,
    plot_2d_samples,
)
from .radial import (
    plot_radial_profile_diagnostics,
)
from .report import (
    plot_mock_fit_report,
)
from .residual import (
    plot_data_model_residual,
)
from .style import (
    finalize_figure,
    make_plot,
    set_plot_style,
    style_axes,
)

__all__ = [
    "set_plot_style",
    "style_axes",
    "make_plot",
    "finalize_figure",
    "corner_plot",
    "erf_sigma_to_percentiles",
    "plot_1d_posterior",
    "plot_2d_samples",
    "plot_data_model_residual",
    "show_log_image",
    "plot_log_data_model_residual",
    "plot_fit_summary",
    "plot_log_image",
    "plot_mock_fit_report",
    "plot_radial_profile_diagnostics",
]