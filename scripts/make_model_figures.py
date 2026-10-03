"""Generate the figures of the Models and Theory sections of the documentation.

Most figures are computed with BayMorph itself, so they show what the code
does. Two are diagrams of conventions, ``model_composition`` and
``geometry``, ``sersic_profiles`` evaluates the Sérsic equations with SciPy,
and ``nested_sampling`` runs a small nested sampler written for the figure.
Every figure is written twice, as ``<name>-light.png`` and
``<name>-dark.png``, for the two colour schemes of the documentation site:
those of ``FIGURES`` to ``docs/models/figures/`` and those of
``THEORY_FIGURES`` to ``docs/theory/figures/``. Run it from the repository
root after a change to a model, a PSF, the background, the noise model, the
effective-PSF rendering or the coupling of several observations::

    python scripts/make_model_figures.py

The figures use the canonical BayMorph plotting style of
``baymorph.plotting.style``: Computer Modern text, panels of 15 x 15 inches
with 35-point text, thick axes and outward ticks with minor ticks. To keep
that text legible at the width of a documentation page, a figure has at most
two panels per row.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import (  # noqa: E402
    LinearSegmentedColormap,
    LogNorm,
    TwoSlopeNorm,
)
from matplotlib.patches import (  # noqa: E402
    Arc,
    Ellipse,
    FancyArrowPatch,
    FancyBboxPatch,
    Patch,
    Rectangle,
)
from matplotlib.ticker import AutoMinorLocator  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402
from scipy.special import gammainc, gammaincinv, gammaln  # noqa: E402

from baymorph.core.correlated_noise import (  # noqa: E402
    CORRELATION_MAX_LAG,
    masked_sky_correlation,
)
from baymorph.core.detection import detect_sources_segmentation  # noqa: E402
from baymorph.core.sky import build_sky_region  # noqa: E402
from baymorph.fitting.hierarchical import (  # noqa: E402
    physical_prior_copula_values,
)
from baymorph.fitting.priors import UniformPrior  # noqa: E402
from baymorph.fitting import (  # noqa: E402
    make_bulge_broken_disk_psf_model,
    make_bulge_disk_bar_psf_model,
    make_bulge_disk_psf_model,
    make_bulge_disk_ring_psf_model,
    make_bulge_disk_spiral_psf_model,
    make_core_sersic_psf_model,
    make_double_sersic_psf_model,
    make_sersic_point_source_psf_model,
    make_single_sersic_psf_model,
)
from baymorph.likelihood import GaussianImageLikelihood  # noqa: E402
from baymorph.models import (  # noqa: E402
    BrokenExponentialComponent,
    CoreSersicComponent,
    FerrersBarComponent,
    GaussianRingComponent,
    GeneralizedSersicComponent,
    LogSpiralDiskComponent,
    PointSourceModel,
    SersicModel,
)
from baymorph.models.isophote import (  # noqa: E402
    generalized_radius,
    major_minor_offsets,
)
from baymorph.observation.background import (  # noqa: E402
    ConstantBackground,
    TiltedPlaneBackground,
)
from baymorph.observation.point_source import (  # noqa: E402
    PSFPointSourceRenderer,
)
from baymorph.pipeline.preparation_steps import estimate_target_sky  # noqa: E402
from baymorph.pipeline.segmentation import (  # noqa: E402
    select_segment_near_position,
)
from baymorph.plotting.style import (  # noqa: E402
    AXES_LINEWIDTH,
    MAJOR_TICK_LENGTH,
    MINOR_TICK_LENGTH,
    TICK_WIDTH,
    canonical_figsize,
    computer_modern_text,
    set_plot_style,
)
from baymorph.rendering.effective_psf import (  # noqa: E402
    SUBPIXEL_FACTOR,
    WINDOW_HALF,
    WINDOW_RADIUS,
    smooth_window,
)
from baymorph.simulation.psf import GaussianPSF  # noqa: E402

OUTPUT_DIRECTORY = Path("docs") / "models" / "figures"
THEORY_DIRECTORY = Path("docs") / "theory" / "figures"

# A panel is 15 inches wide; at 50 dots per inch a two-panel figure is
# 1500 pixels wide.
DPI = 50

# Line widths and text sizes in points, for panels of 15 x 15 inches.
LINE = 4.0
THIN = 2.5
LEGEND_SIZE = 28
NOTE_SIZE = 28
MARKER = 16.0


@dataclass(frozen=True)
class Theme:
    """Colours of one colour scheme, from the validated reference palette."""

    name: str
    surface: str
    ink: str
    secondary: str
    muted: str
    grid: str
    axis: str
    series: tuple[str, ...]
    image_ramp: tuple[str, ...]
    diverging: tuple[str, str, str]


LIGHT = Theme(
    name="light",
    surface="#fcfcfb",
    ink="#0b0b0b",
    secondary="#52514e",
    muted="#898781",
    grid="#e1e0d9",
    axis="#c3c2b7",
    series=(
        "#2a78d6", "#eb6834", "#1baf7a", "#eda100",
        "#e87ba4", "#008300", "#4a3aa7", "#e34948",
    ),
    image_ramp=(
        "#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5",
        "#256abf", "#184f95", "#0d366b",
    ),
    diverging=("#2a78d6", "#f0efec", "#e34948"),
)

DARK = Theme(
    name="dark",
    surface="#1a1a19",
    ink="#ffffff",
    secondary="#c3c2b7",
    muted="#898781",
    grid="#2c2c2a",
    axis="#383835",
    series=(
        "#3987e5", "#d95926", "#199e70", "#c98500",
        "#d55181", "#008300", "#9085e9", "#e66767",
    ),
    image_ramp=(
        "#1a1a19", "#0d366b", "#184f95", "#256abf",
        "#3987e5", "#86b6ef", "#cde2fb",
    ),
    diverging=("#3987e5", "#383835", "#e66767"),
)

THEMES = (LIGHT, DARK)

# Compact components (bulge, point source) wear slot 2, extended ones (disc,
# Sérsic) slot 1 and the structures of a disc (bar, ring) slot 3 in every
# figure, so a colour keeps one meaning.
ROLE_SLOT = {
    "bulge": 1,
    "point_source": 1,
    "disk": 0,
    "sersic": 0,
    "bar": 2,
    "ring": 2,
}


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------


def apply_style(theme: Theme) -> None:
    """Set the BayMorph style with the colours of one colour scheme."""
    set_plot_style()
    plt.rcParams.update(
        {
            "text.color": theme.ink,
            "axes.labelcolor": theme.ink,
            "axes.titlecolor": theme.ink,
            "axes.edgecolor": theme.ink,
            "axes.facecolor": theme.surface,
            "axes.titlesize": 35,
            "axes.titlepad": 18,
            "axes.grid": False,
            "xtick.color": theme.ink,
            "ytick.color": theme.ink,
            "figure.facecolor": theme.surface,
            "savefig.facecolor": theme.surface,
            "lines.linewidth": LINE,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
            "legend.frameon": False,
            "legend.fontsize": LEGEND_SIZE,
            "legend.handlelength": 1.6,
        }
    )


def cm(text: str) -> str:
    """Return text whose accents the Computer Modern font can draw."""
    return computer_modern_text(text)


def image_colormap(theme: Theme) -> LinearSegmentedColormap:
    """Return the sequential one-hue ramp of the scheme, low to high."""
    return LinearSegmentedColormap.from_list(
        f"baymorph-{theme.name}", theme.image_ramp
    )


def diverging_colormap(theme: Theme) -> LinearSegmentedColormap:
    """Return the blue-to-red ramp with a neutral midpoint."""
    return LinearSegmentedColormap.from_list(
        f"baymorph-diverging-{theme.name}", theme.diverging
    )


def canonical_axes(ax) -> None:
    """Draw outward ticks, with minor ticks on the linear axes.

    Call it after the scales are set: a logarithmic axis keeps its own minor
    ticks.
    """
    ax.tick_params(
        which="major",
        length=MAJOR_TICK_LENGTH,
        width=TICK_WIDTH,
        direction="out",
    )
    ax.tick_params(
        which="minor",
        length=MINOR_TICK_LENGTH,
        width=TICK_WIDTH,
        direction="out",
    )
    if ax.get_xscale() == "linear":
        ax.xaxis.set_minor_locator(AutoMinorLocator())
    if ax.get_yscale() == "linear":
        ax.yaxis.set_minor_locator(AutoMinorLocator())


def panels(nrows: int, ncols: int, **kwargs):
    """Return a figure of canonical panels and its grid of axes."""
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=canonical_figsize(nrows, ncols),
        squeeze=False,
        **kwargs,
    )
    fig.subplots_adjust(wspace=0.32, hspace=0.38)
    return fig, axes


def show_image(ax, image, theme: Theme, norm, extent=None) -> None:
    """Draw an image with its origin at the bottom left, as BayMorph does."""
    ax.imshow(
        np.clip(image, norm.vmin, None) if isinstance(norm, LogNorm) else image,
        origin="lower",
        cmap=image_colormap(theme),
        norm=norm,
        interpolation="nearest",
        extent=extent,
    )
    ax.set_xlabel(r"$x$ [pix]")
    ax.set_ylabel(r"$y$ [pix]")
    canonical_axes(ax)


def reference_line(ax, theme: Theme, *, x=None, y=None) -> None:
    """Draw a dashed reference line."""
    if x is not None:
        ax.axvline(x, color=theme.muted, linewidth=THIN, linestyle="--", zorder=1)
    if y is not None:
        ax.axhline(y, color=theme.muted, linewidth=THIN, linestyle="--", zorder=1)


def side_colorbar(fig, ax, handle, theme: Theme, label: str) -> None:
    """Draw a colorbar as tall as an image panel, on its right."""
    ax.apply_aspect()
    position = ax.get_position()
    colorbar_axis = fig.add_axes(
        (position.x1 + 0.02, position.y0, 0.015, position.height)
    )
    bar = fig.colorbar(handle, cax=colorbar_axis)
    bar.outline.set_edgecolor(theme.ink)
    bar.outline.set_linewidth(AXES_LINEWIDTH)
    bar.ax.tick_params(which="both", direction="out", width=TICK_WIDTH,
                       colors=theme.ink)
    bar.ax.tick_params(which="major", length=MAJOR_TICK_LENGTH)
    bar.set_label(label, labelpad=24)


def save(fig, name: str, theme: Theme, directory: Path) -> Path:
    """Write one figure and close it."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}-{theme.name}.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    return path


def radial_profile(image, x0: float, y0: float, rmax: float):
    """Return the mean of an image in circular annuli one pixel wide."""
    yy, xx = np.indices(image.shape, dtype=float)
    radius = np.hypot(xx - x0, yy - y0)
    edges = np.arange(0.0, rmax + 1.0, 1.0)
    centres = 0.5 * (edges[:-1] + edges[1:])
    profile = np.array(
        [
            image[(radius >= lo) & (radius < hi)].mean()
            for lo, hi in zip(edges[:-1], edges[1:], strict=True)
        ]
    )
    return centres, profile


def signed(value: float) -> str:
    """Write a signed number in mathematical text, with a true minus."""
    return "0" if value == 0.0 else f"{value:+.1f}"


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------


def forward_model(theme: Theme) -> plt.Figure:
    """From the intrinsic morphology to an observed image."""
    shape = (81, 81)
    parameters = dict(
        flux=400.0, re=5.0, n=3.0, q=0.6, theta=np.deg2rad(30.0),
        x0=40.3, y0=39.8,
    )
    psf = GaussianPSF(fwhm=5.0)
    intrinsic = SersicModel(**parameters).render(shape)
    convolved = make_single_sersic_psf_model(**parameters, psf=psf).render(shape)
    background = TiltedPlaneBackground(
        level=0.03, gradient_x=0.0006, gradient_y=-0.0004
    ).render(shape)
    expected = convolved + background
    data = expected + np.random.default_rng(7).normal(0.0, 0.012, shape)

    norm = LogNorm(vmin=0.01, vmax=float(convolved.max()))
    images = (
        (intrinsic, r"(1) Intrinsic, $I$"),
        (convolved, r"(2) After the PSF, $P \circledast I$"),
        (expected, r"(3) Plus background, $\mu = P \circledast I + B$"),
        (data, r"(4) Observed, $D = \mu + \varepsilon$"),
    )
    fig, axes = panels(2, 2)
    for ax, (image, title) in zip(axes.flat, images, strict=True):
        show_image(ax, image, theme, norm)
        ax.set_title(title)
    return fig


def model_composition(theme: Theme) -> plt.Figure:
    """Which components each morphology model is made of."""
    fig, ax = plt.subplots(figsize=(30.0, 11.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(1.0, 5.75)
    ax.axis("off")

    def box(x, y, w, h, title, detail=None, emphasis=False):
        ax.add_patch(
            FancyBboxPatch(
                (x, y), w, h,
                boxstyle="round,pad=0.02,rounding_size=0.12",
                facecolor=theme.grid if emphasis else theme.surface,
                edgecolor=theme.ink,
                linewidth=THIN,
            )
        )
        if detail is None:
            ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                    color=theme.ink, fontsize=33)
            return
        ax.text(x + w / 2, y + h / 2 + 0.15, title, ha="center", va="center",
                color=theme.ink, fontsize=33)
        ax.text(x + w / 2, y + h / 2 - 0.19, detail, ha="center", va="center",
                color=theme.secondary, fontsize=27)

    def link(start, end):
        ax.add_patch(
            FancyArrowPatch(
                start, end, arrowstyle="-|>", mutation_scale=30,
                color=theme.muted, linewidth=THIN, shrinkA=0, shrinkB=0,
            )
        )

    ax.text(1.3, 5.45, "Components", ha="center", color=theme.secondary)
    ax.text(5.0, 5.45, "Morphology models", ha="center", color=theme.secondary)
    ax.text(8.6, 5.45, "Observation", ha="center", color=theme.secondary)

    box(0.15, 3.85, 2.3, 0.85, cm("Sérsic profile"))
    box(0.15, 2.55, 2.3, 0.85, "Exponential disc", cm("Sérsic with $n = 1$"))
    box(0.15, 1.25, 2.3, 0.85, "Point source", "unresolved")

    models = (
        ("single\\_sersic", cm("one Sérsic"), 4.25),
        ("double\\_sersic", cm("Sérsic + Sérsic"), 3.25),
        ("bulge\\_disk", cm("Sérsic bulge + exponential disc"), 2.25),
        ("sersic\\_point\\_source", cm("Sérsic + point source"), 1.25),
    )
    for name, detail, y in models:
        box(3.55, y, 2.9, 0.78, rf"$\mathtt{{{name}}}$", detail, emphasis=True)

    sersic_out = (2.45, 4.27)
    for _, _, y in models:
        link(sersic_out, (3.55, y + 0.39))
    link((2.45, 2.97), (3.55, 2.64))
    link((2.45, 1.67), (3.55, 1.64))

    box(7.35, 2.15, 2.5, 1.65,
        "convolved with the PSF\n+ residual background\n= expected image")
    for _, _, y in models:
        link((6.45, y + 0.39), (7.35, 2.97))
    return fig


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------


def geometry(theme: Theme) -> plt.Figure:
    """The coordinate and shape conventions of a component."""
    fig, ax = plt.subplots(figsize=(15.0, 12.7))
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 8.5)
    ax.set_aspect("equal")
    for value in np.arange(-0.5, 11.0, 1.0):
        ax.axvline(value, color=theme.grid, linewidth=THIN, zorder=0)
    for value in np.arange(-0.5, 9.0, 1.0):
        ax.axhline(value, color=theme.grid, linewidth=THIN, zorder=0)
    ax.add_patch(
        Rectangle(
            (1.5, 0.5), 1.0, 1.0,
            facecolor=theme.grid, edgecolor=theme.muted, linewidth=THIN,
            zorder=1,
        )
    )
    ax.plot(2.0, 1.0, "o", color=theme.muted, markersize=10, zorder=2)
    ax.annotate(
        r"pixel [1, 2], centred at $(x, y) = (2, 1)$",
        xy=(2.5, 1.0), xytext=(3.3, 0.15),
        color=theme.ink, fontsize=NOTE_SIZE,
        arrowprops=dict(arrowstyle="-", color=theme.muted, linewidth=THIN),
    )

    x0, y0, re, q, theta = 5.4, 4.6, 3.3, 0.55, np.deg2rad(30.0)
    colour = theme.series[0]
    ax.add_patch(
        Ellipse(
            (x0, y0), 2 * re, 2 * q * re, angle=np.rad2deg(theta),
            fill=False, edgecolor=colour, linewidth=LINE + 1, zorder=3,
        )
    )
    major = (x0 + re * np.cos(theta), y0 + re * np.sin(theta))
    minor = (
        x0 - q * re * np.sin(theta), y0 + q * re * np.cos(theta)
    )
    ax.plot([x0, major[0]], [y0, major[1]], color=colour, zorder=3)
    ax.plot([x0, minor[0]], [y0, minor[1]], color=colour, zorder=3)
    ax.plot([x0, x0 + 2.6], [y0, y0], color=theme.muted, linewidth=THIN,
            linestyle="--", zorder=2)
    ax.add_patch(
        Arc(
            (x0, y0), 2.4, 2.4, theta1=0.0, theta2=np.rad2deg(theta),
            color=theme.ink, linewidth=THIN, zorder=3,
        )
    )
    ax.plot(
        x0, y0, "o", color=colour, markersize=MARKER,
        markeredgecolor=theme.surface, markeredgewidth=3, zorder=4,
    )
    ax.text(x0 - 0.15, y0 - 0.5, r"$(x_0, y_0)$", ha="right", color=theme.ink)
    ax.text(
        x0 + 0.72 * (major[0] - x0) + 0.25, y0 + 0.72 * (major[1] - y0) - 0.15,
        r"$R_e$", color=theme.ink,
    )
    ax.text(
        0.5 * (x0 + minor[0]) - 1.15, 0.5 * (y0 + minor[1]) + 0.1,
        r"$q\,R_e$", color=theme.ink,
    )
    ax.text(x0 + 1.35, y0 + 0.1, r"$\theta$", color=theme.ink)
    ax.set_xticks(range(0, 11, 2))
    ax.set_yticks(range(0, 9, 2))
    ax.set_xlabel(r"$x$ [pix]")
    ax.set_ylabel(r"$y$ [pix]")
    canonical_axes(ax)
    return fig


SERIES_INDICES = (0.5, 1.0, 2.0, 4.0, 6.0)


def sersic_profiles(theme: Theme) -> plt.Figure:
    """Profiles and enclosed light of the Sérsic profile."""
    fig, axes = panels(1, 2)
    left, right = axes[0]
    radius = np.linspace(0.0, 4.0, 400)
    log_radius = np.logspace(-2, 1, 400)
    for index, colour in zip(SERIES_INDICES, theme.series, strict=False):
        bn = float(gammaincinv(2 * index, 0.5))
        profile = np.exp(-bn * (radius ** (1 / index) - 1.0))
        left.plot(radius, profile, color=colour, label=rf"$n = {index:g}$")
        enclosed = gammainc(2 * index, bn * log_radius ** (1 / index))
        right.plot(log_radius, enclosed, color=colour)
    left.set_yscale("log")
    left.set_ylim(1e-6, 3e3)
    left.set_xlim(0, 4)
    left.set_xlabel(r"$R / R_e$")
    left.set_ylabel(r"$I(R) / I_e$")
    left.set_title("Surface brightness")
    left.legend(loc="lower left")
    right.set_xscale("log")
    right.set_xlim(1e-2, 10)
    right.set_ylim(0, 1.02)
    reference_line(right, theme, x=1.0, y=0.5)
    right.text(
        1.15, 0.05, r"$R = R_e$ holds" "\nhalf the light",
        color=theme.secondary, fontsize=NOTE_SIZE,
    )
    right.set_xlabel(r"$R / R_e$")
    right.set_ylabel(r"$L(<R) / F$")
    right.set_title("Enclosed light")
    for ax in (left, right):
        canonical_axes(ax)
    return fig


def point_source_placement(theme: Theme) -> plt.Figure:
    """A point source is the PSF placed at a sub-pixel position."""
    shape = (11, 11)
    renderer = PSFPointSourceRenderer(GaussianPSF(fwhm=2.5))
    offsets = (0.0, 0.25, 0.5)
    fig, axes = panels(2, 2)
    norm = LogNorm(vmin=1e-4, vmax=0.17)
    profile_axis = axes[1, 1]
    for ax, offset, colour in zip(
        axes.flat[:3], offsets, theme.series, strict=False
    ):
        source = PointSourceModel(flux=1.0, x0=5.0 + offset, y0=5.0)
        image = renderer.render(source, shape)
        show_image(ax, image, theme, norm)
        ax.plot(
            source.x0, source.y0, "o", color=theme.series[1],
            markersize=MARKER, markeredgecolor=theme.surface,
            markeredgewidth=3,
        )
        ax.set_title(rf"Offset ${offset:g}$ px")
        profile_axis.step(
            np.arange(shape[1]), image[5], where="mid", color=colour,
            label=rf"offset ${offset:g}$ px",
        )
    profile_axis.set_xlabel(r"$x$ [pix]")
    profile_axis.set_ylabel("Pixel value")
    profile_axis.set_title(r"The row $y = 5$")
    profile_axis.set_xlim(0, 10)
    profile_axis.set_ylim(bottom=0)
    profile_axis.legend(loc="upper left")
    canonical_axes(profile_axis)
    return fig


# ---------------------------------------------------------------------------
# Morphology models
# ---------------------------------------------------------------------------


def single_sersic_parameters(theme: Theme) -> plt.Figure:
    """What each parameter of a Sérsic profile does to the image.

    Each row is one strip of four images in which one parameter changes and
    the others keep the values of ``base``.
    """
    shape = (61, 61)
    psf = GaussianPSF(fwhm=2.5)
    base = dict(
        flux=100.0, re=7.0, n=2.0, q=0.7, theta=np.deg2rad(30.0),
        x0=30.0, y0=30.0,
    )
    rows = (
        ("n", r"$n$", (0.5, 1.0, 2.5, 5.0), lambda v: rf"$n = {v:g}$"),
        ("re", r"$R_e$", (3.0, 6.0, 10.0, 16.0), lambda v: rf"$R_e = {v:g}$ px"),
        ("q", r"$q$", (1.0, 0.8, 0.6, 0.4), lambda v: rf"$q = {v:g}$"),
        (
            "theta", r"$\theta$", (0.0, 45.0, 90.0, 135.0),
            lambda v: rf"$\theta = {v:g}^\circ$",
        ),
    )
    fig, axes = plt.subplots(4, 1, figsize=(30.0, 35.0))
    fig.subplots_adjust(hspace=0.22)
    norm = LogNorm(vmin=0.01, vmax=12.0)
    gap = np.full((shape[0], 3), np.nan)
    for ax, (key, label, values, caption) in zip(axes, rows, strict=True):
        tiles = []
        for value in values:
            parameters = dict(base)
            parameters[key] = np.deg2rad(value) if key == "theta" else value
            image = make_single_sersic_psf_model(**parameters, psf=psf).render(shape)
            tiles.extend((np.clip(image, norm.vmin, None), gap))
        strip = np.hstack(tiles[:-1])
        colormap = image_colormap(theme).copy()
        colormap.set_bad(theme.surface)
        ax.imshow(strip, origin="lower", cmap=colormap, norm=norm,
                  interpolation="nearest")
        step = shape[1] + gap.shape[1]
        ax.set_xticks([column * step + 0.5 * (shape[1] - 1) for column in range(4)])
        ax.set_xticklabels([caption(value) for value in values])
        ax.set_yticks([])
        ax.set_ylabel(label, rotation=0, labelpad=40, fontsize=40)
        ax.tick_params(which="major", length=0, pad=12)
        for spine in ax.spines.values():
            spine.set_visible(False)
    return fig


def decomposition(theme: Theme, model, names, labels, title_total, rmax):
    """Images of the components and the total, with their radial profiles.

    Two components fill a 2 x 2 grid with the profiles in its last panel;
    three components fill two rows of images, and the profiles span a third.
    """
    shape = (101, 101)
    components = model.render_components(shape)
    total = sum(components[name] for name in names)
    norm = LogNorm(vmin=1e-4 * float(total.max()), vmax=float(total.max()))
    images = [
        (components[name], label)
        for name, label in zip(names, labels, strict=True)
    ]
    images.append((total, title_total))

    if len(images) == 3:
        fig, axes = panels(2, 2)
        image_axes = list(axes.flat[:3])
        profile_axis = axes[1, 1]
    else:
        fig = plt.figure(figsize=canonical_figsize(3, 2))
        grid = fig.add_gridspec(3, 2, wspace=0.32, hspace=0.38)
        image_axes = [fig.add_subplot(grid[row, column])
                      for row in range(2) for column in range(2)]
        profile_axis = fig.add_subplot(grid[2, :])

    for ax, (image, label) in zip(image_axes, images, strict=True):
        show_image(ax, image, theme, norm)
        ax.set_title(label)

    for name, label in zip(names, labels, strict=True):
        colour = theme.series[ROLE_SLOT[name]]
        r, profile = radial_profile(components[name], 50.0, 50.0, rmax)
        profile_axis.plot(r, profile, color=colour, label=label)
    r, profile = radial_profile(total, 50.0, 50.0, rmax)
    profile_axis.plot(r, profile, color=theme.ink, linewidth=LINE + 1,
                      label="Total")
    profile_axis.set_yscale("log")
    profile_axis.set_ylim(1e-4 * float(profile.max()), 2.0 * float(profile.max()))
    profile_axis.set_xlim(0, rmax)
    profile_axis.set_xlabel(r"$R$ [pix]")
    profile_axis.set_ylabel("Mean intensity")
    profile_axis.set_title("Radial profiles")
    profile_axis.legend(loc="upper right")
    canonical_axes(profile_axis)
    return fig


def double_sersic_decomposition(theme: Theme) -> plt.Figure:
    """The two labelled components of a double Sérsic model."""
    model = make_double_sersic_psf_model(
        flux_bulge=40.0, re_bulge=3.0, n_bulge=4.0, q_bulge=0.85,
        theta_bulge=np.deg2rad(20.0),
        flux_disk=60.0, re_disk=14.0, n_disk=1.0, q_disk=0.5,
        theta_disk=np.deg2rad(35.0),
        x0=50.0, y0=50.0, psf=GaussianPSF(fwhm=2.5),
    )
    return decomposition(
        theme, model, ("bulge", "disk"), ("Bulge component", "Disk component"),
        "Total", 48,
    )


def sersic_point_source_decomposition(theme: Theme) -> plt.Figure:
    """A Sérsic profile and an unresolved source at the same centre."""
    model = make_sersic_point_source_psf_model(
        flux_sersic=80.0, re=8.0, n=2.0, q=0.7, theta=np.deg2rad(30.0),
        flux_point_source=20.0, x0=50.0, y0=50.0, psf=GaussianPSF(fwhm=2.5),
    )
    return decomposition(
        theme, model, ("sersic", "point_source"),
        (cm("Sérsic component"), "Point source"), "Total", 40,
    )


def bulge_disk_decomposition(theme: Theme) -> plt.Figure:
    """A Sérsic bulge and an exponential disc."""
    model = make_bulge_disk_psf_model(
        flux_bulge=30.0, re_bulge=3.0, n_bulge=3.5, q_bulge=0.9,
        theta_bulge=np.deg2rad(40.0),
        flux_disk=70.0, re_disk=16.0, q_disk=0.45, theta_disk=np.deg2rad(40.0),
        x0=50.0, y0=50.0, psf=GaussianPSF(fwhm=2.5),
    )
    return decomposition(
        theme, model, ("bulge", "disk"), (r"Bulge, $n = 3.5$", "Exponential disc"),
        r"Total, $B/T = 0.3$", 48,
    )


# ---------------------------------------------------------------------------
# PSF and background
# ---------------------------------------------------------------------------


def gaussian_psf(theme: Theme) -> plt.Figure:
    """The Gaussian kernel and its profile."""
    psf = GaussianPSF(fwhm=3.0)
    kernel = psf.kernel()
    fig, axes = panels(1, 2)
    ax = axes[0, 0]
    show_image(ax, kernel, theme, LogNorm(vmin=1e-6, vmax=float(kernel.max())))
    ax.set_title(r"Kernel, $25 \times 25$ px")
    ax = axes[0, 1]
    centre = kernel.shape[0] // 2
    yy, xx = np.indices(kernel.shape)
    radius = np.hypot(xx - centre, yy - centre).ravel()
    order = np.argsort(radius)
    ax.plot(
        radius[order], kernel.ravel()[order] / kernel.max(), "o",
        color=theme.series[0], markersize=10, markeredgewidth=0,
        label="Kernel pixels",
    )
    fine = np.linspace(0, 8, 300)
    ax.plot(
        fine, np.exp(-0.5 * fine**2 / psf.sigma**2), color=theme.ink,
        label="Gaussian",
    )
    reference_line(ax, theme, y=0.5)
    ax.annotate(
        "", xy=(0.0, 0.5), xytext=(psf.fwhm / 2, 0.5),
        arrowprops=dict(arrowstyle="<->", color=theme.ink, linewidth=THIN,
                        mutation_scale=30),
    )
    ax.text(0.15, 0.56, "FWHM / 2", color=theme.ink, fontsize=NOTE_SIZE)
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel(r"$r$ [pix]")
    ax.set_ylabel("Normalized value")
    ax.set_title("Profile")
    ax.legend(loc="upper right")
    canonical_axes(ax)
    return fig


def effective_psf_window(theme: Theme) -> plt.Figure:
    """How the effective-PSF rendering splits a profile."""
    fig, axes = panels(1, 2)
    ax = axes[0, 0]
    radius = np.linspace(0.0, 16.0, 800)
    window = smooth_window(radius, np.zeros_like(radius), 0.0, 0.0)
    inner, outer = theme.series[0], theme.series[1]
    ax.plot(radius, window, color=inner, label=r"$w$: fine cells")
    ax.plot(radius, 1.0 - window, color=outer, label=r"$1 - w$: pixel centres")
    reference_line(ax, theme, x=WINDOW_RADIUS)
    reference_line(ax, theme, x=WINDOW_HALF)
    ax.text(WINDOW_RADIUS + 0.25, 0.5, "7 px", color=theme.ink,
            fontsize=NOTE_SIZE)
    ax.text(WINDOW_HALF - 0.25, 0.5, "cut, 14 px", color=theme.ink,
            fontsize=NOTE_SIZE, ha="right")
    ax.set_xlim(0, 16)
    ax.set_ylim(-0.02, 1.3)
    ax.set_xlabel("Distance from the centre [pix]")
    ax.set_ylabel("Weight")
    ax.set_title("The two parts of a profile")
    ax.legend(loc="upper right")
    canonical_axes(ax)

    ax = axes[0, 1]
    half = 16
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1].astype(float)
    weights = smooth_window(xx, yy, 0.0, 0.0)
    ax.imshow(
        weights, origin="lower", cmap=image_colormap(theme),
        extent=(-half - 0.5, half + 0.5, -half - 0.5, half + 0.5),
        interpolation="nearest",
    )
    angle = np.linspace(0.0, 2.0 * np.pi, 400)
    for radius_value, colour in ((WINDOW_RADIUS, theme.ink), (WINDOW_HALF, theme.muted)):
        ax.plot(
            radius_value * np.cos(angle), radius_value * np.sin(angle),
            color=colour, linewidth=THIN,
        )
    ax.text(0.0, 0.0, rf"cells ${SUBPIXEL_FACTOR}\times$" "\nfiner",
            ha="center", va="center", color=theme.surface, fontsize=NOTE_SIZE)
    ax.text(0.0, -11.0, "pixel centres", ha="center", va="center",
            color=theme.ink, fontsize=NOTE_SIZE)
    ax.set_xlabel(r"$\Delta x$ [pix]")
    ax.set_ylabel(r"$\Delta y$ [pix]")
    ax.set_title(r"The window $w$ about the centre")
    canonical_axes(ax)
    return fig


def background_models(theme: Theme) -> plt.Figure:
    """The constant and the plane residual backgrounds."""
    shape = (101, 101)
    constant = ConstantBackground(level=0.3).render(shape)
    plane = TiltedPlaneBackground(
        level=0.3, gradient_x=0.008, gradient_y=-0.005
    ).render(shape)
    limit = float(np.abs(plane).max())
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-limit, vmax=limit)
    fig, axes = panels(1, 2)
    for ax, image, title in (
        (axes[0, 0], constant, "Constant"),
        (axes[0, 1], plane, "Plane"),
    ):
        handle = ax.imshow(
            image, origin="lower", cmap=diverging_colormap(theme), norm=norm,
        )
        ax.set_xlabel(r"$x$ [pix]")
        ax.set_ylabel(r"$y$ [pix]")
        ax.set_title(title)
        canonical_axes(ax)
    centre = 0.5 * (shape[0] - 1)
    ax = axes[0, 1]
    ax.plot(
        centre, centre, "+", color=theme.ink, markersize=30, markeredgewidth=4,
    )
    ax.annotate(
        r"$(x_{\rm ref}, y_{\rm ref})$: level $b$", xy=(centre, centre),
        xytext=(centre - 44, centre + 34), color=theme.ink, fontsize=NOTE_SIZE,
        arrowprops=dict(arrowstyle="-", color=theme.ink, linewidth=THIN),
    )
    direction = np.array([0.008, -0.005])
    direction = 22.0 * direction / np.linalg.norm(direction)
    ax.annotate(
        "", xy=(centre + direction[0], centre + direction[1]),
        xytext=(centre, centre),
        arrowprops=dict(arrowstyle="-|>", color=theme.ink, linewidth=LINE,
                        mutation_scale=40),
    )
    side_colorbar(fig, ax, handle, theme, "Residual background")
    return fig


def sky_region(theme: Theme) -> plt.Figure:
    """The plateau of a galaxy and the sky region beyond it.

    The image is made, detected and measured with the functions of the
    preparation: the target segment, the plateau of its elliptical radial
    profile and the sky region beyond the plateau radius plus 5 px.
    """
    shape = (201, 201)
    psf = GaussianPSF(fwhm=3.0)
    galaxy = make_single_sersic_psf_model(
        flux=3.0e4, re=10.0, n=2.0, q=0.7, theta=np.deg2rad(30.0),
        x0=100.0, y0=100.0, psf=psf,
    ).render(shape)
    neighbour = make_single_sersic_psf_model(
        flux=3.0e3, re=3.0, n=1.0, q=0.8, theta=0.0, x0=160.0, y0=45.0,
        psf=psf,
    ).render(shape)
    sky, noise = 100.0, 5.0
    image = galaxy + neighbour + sky + np.random.default_rng(11).normal(
        0.0, noise, shape
    )
    detection = detect_sources_segmentation(image, n_sigma=3.0, n_pixels=20)
    segmentation = detection.segmentation.data
    label = select_segment_near_position(detection.segmentation, x=100.0, y=100.0)
    target = segmentation == label
    contaminants = (segmentation > 0) & ~target
    geometry, profile, plateau = estimate_target_sky(
        image, target, mask=contaminants, min_radius=0.0, window_size=5,
        sigma_threshold=1.0, outer_fraction=0.25, min_coverage=0.5,
    )
    region = build_sky_region(shape, geometry, plateau, margin=5.0,
                              mask=contaminants)

    fig, axes = panels(1, 2)
    ax = axes[0, 0]
    show_image(ax, image - sky, theme,
               LogNorm(vmin=0.3 * noise, vmax=float(galaxy.max())))
    overlay = np.zeros(shape + (4,))
    overlay[region.mask] = matplotlib.colors.to_rgba(theme.series[2], 0.28)
    overlay[contaminants] = matplotlib.colors.to_rgba(theme.muted, 0.9)
    ax.imshow(overlay, origin="lower", interpolation="nearest")
    for radius, style in ((plateau.radius, "-"), (region.inner_radius, "--")):
        ax.add_patch(
            Ellipse(
                (geometry.x0, geometry.y0), 2.0 * radius,
                2.0 * geometry.q * radius, angle=np.rad2deg(geometry.theta),
                fill=False, edgecolor=theme.ink, linewidth=THIN,
                linestyle=style,
            )
        )
    ax.set_title("The sky region")
    ax.legend(
        handles=(
            Patch(facecolor=theme.series[2], alpha=0.5, label="sky region"),
            Patch(facecolor=theme.muted, label="masked source"),
        ),
        loc="upper left",
    )

    ax = axes[0, 1]
    ax.plot(profile.radius, profile.profile, color=theme.series[0])
    reference_line(ax, theme, x=plateau.radius, y=plateau.background_level)
    ax.axvspan(region.inner_radius, float(profile.radius.max()),
               color=theme.series[2], alpha=0.18, linewidth=0)
    ax.text(plateau.radius + 2.0, 0.6 * float(profile.profile.max()),
            "plateau", color=theme.ink, fontsize=NOTE_SIZE)
    ax.text(float(profile.radius.max()) - 2.0, plateau.background_level * 1.25,
            "sky level", color=theme.ink, fontsize=NOTE_SIZE, ha="right")
    ax.set_yscale("log")
    ax.set_xlim(0, float(profile.radius.max()))
    ax.set_xlabel("Elliptical radius [pix]")
    ax.set_ylabel("Median intensity")
    ax.set_title("The radial profile")
    canonical_axes(ax)
    return fig


def noise_variance(theme: Theme) -> plt.Figure:
    """The noise of a pixel: the sky and the Poisson noise of the galaxy."""
    sky_scale = 20.0
    fig, axes = panels(1, 2)
    ax = axes[0, 0]
    model = np.logspace(0.0, 5.0, 400)
    for gain, colour in zip((0.5, 1.0, 4.0), theme.series, strict=False):
        ax.plot(model, np.sqrt(sky_scale**2 + model / gain), color=colour,
                label=rf"$g = {gain:g}$")
    reference_line(ax, theme, y=sky_scale)
    ax.text(1.5, sky_scale * 1.12, "sky noise $s$", color=theme.ink,
            fontsize=NOTE_SIZE)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1.0, 1.0e5)
    ax.set_ylim(10.0, 600.0)
    ax.set_xlabel(r"Galaxy model $m$ [image units]")
    ax.set_ylabel(r"Noise $\sigma = \sqrt{s^2 + m/g}$")
    ax.set_title("The noise of one pixel")
    ax.legend(loc="upper left")
    canonical_axes(ax)

    shape = (101, 101)
    galaxy = make_single_sersic_psf_model(
        flux=2.0e5, re=8.0, n=2.5, q=0.7, theta=np.deg2rad(30.0),
        x0=50.0, y0=50.0, psf=GaussianPSF(fwhm=2.5),
    ).render(shape)
    sigma = np.sqrt(sky_scale**2 + np.clip(galaxy, 0.0, None) / 1.0)
    ax = axes[0, 1]
    handle = ax.imshow(sigma, origin="lower", cmap=image_colormap(theme),
                       interpolation="nearest")
    ax.set_xlabel(r"$x$ [pix]")
    ax.set_ylabel(r"$y$ [pix]")
    ax.set_title(r"The noise of each pixel, $g = 1$")
    canonical_axes(ax)
    side_colorbar(fig, ax, handle, theme, r"$\sigma$ [image units]")
    return fig


def correlated_noise(theme: Theme) -> plt.Figure:
    """White and correlated noise, their correlation and the neighbours."""
    rng = np.random.default_rng(3)
    white = rng.normal(0.0, 1.0, (256, 256))
    correlated = gaussian_filter(white, sigma=0.9)
    correlated /= correlated.std()
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-3.0, vmax=3.0)
    fig, axes = panels(2, 2)
    for ax, image, title in (
        (axes[0, 0], white, "White noise"),
        (axes[0, 1], correlated, "Correlated noise"),
    ):
        ax.imshow(image[:48, :48], origin="lower", cmap=diverging_colormap(theme),
                  norm=norm, interpolation="nearest")
        ax.set_xlabel(r"$x$ [pix]")
        ax.set_ylabel(r"$y$ [pix]")
        ax.set_title(title)
        canonical_axes(ax)

    ax = axes[1, 0]
    everywhere = np.ones(white.shape, dtype=bool)
    lags = np.arange(CORRELATION_MAX_LAG + 1)
    for image, colour, label in (
        (white, theme.series[0], "white"),
        (correlated, theme.series[1], "correlated"),
    ):
        kernel, _ = masked_sky_correlation(
            image, everywhere, max_lag=CORRELATION_MAX_LAG
        )
        ax.plot(lags, kernel[CORRELATION_MAX_LAG, CORRELATION_MAX_LAG:], "o-",
                color=colour, markersize=MARKER * 0.7, label=label)
    reference_line(ax, theme, y=0.0)
    ax.set_xlim(-0.3, CORRELATION_MAX_LAG + 0.3)
    ax.set_ylim(-0.15, 1.05)
    ax.set_xlabel(r"Lag along $x$ [pix]")
    ax.set_ylabel("Correlation")
    ax.set_title("Correlation with distance")
    ax.legend(loc="upper right")
    canonical_axes(ax)

    # The factorization visits the pixels row by row and conditions each on
    # its 24 nearest predecessors, the most recent first among equals.
    ax = axes[1, 1]
    size, wanted = 11, 24
    centre = size // 2
    yy, xx = np.indices((size, size))
    index = yy * size + xx
    current = centre * size + centre
    previous = index < current
    distance = (xx - centre) ** 2 + (yy - centre) ** 2
    order = np.lexsort((-index[previous], distance[previous]))
    neighbours = np.zeros((size, size), dtype=bool)
    neighbours.flat[index[previous][order[:wanted]]] = True
    for (y, x), _ in np.ndenumerate(index):
        if neighbours[y, x]:
            colour = theme.series[0]
        elif index[y, x] == current:
            colour = theme.series[1]
        elif previous[y, x]:
            colour = theme.grid
        else:
            colour = theme.surface
        ax.add_patch(
            Rectangle((x - centre - 0.5, y - centre - 0.5), 1.0, 1.0,
                      facecolor=colour, edgecolor=theme.muted, linewidth=1.5)
        )
    ax.set_xlim(-centre - 0.5, centre + 0.5)
    ax.set_ylim(-centre - 0.5, centre + 0.5)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$\Delta x$ [pix]")
    ax.set_ylabel(r"$\Delta y$ [pix]")
    ax.set_title(r"The $24$ neighbours of a pixel")
    ax.legend(
        handles=(
            Patch(facecolor=theme.series[1], label="pixel"),
            Patch(facecolor=theme.series[0], label="its neighbours"),
            Patch(facecolor=theme.grid, edgecolor=theme.muted,
                  label="earlier pixels"),
        ),
        loc="upper left", bbox_to_anchor=(0.0, -0.18), ncols=3,
        fontsize=24, columnspacing=1.0,
    )
    canonical_axes(ax)
    return fig


# ---------------------------------------------------------------------------
# Degeneracies
# ---------------------------------------------------------------------------


def sersic_degeneracy(theme: Theme) -> plt.Figure:
    """A Sérsic of another index matched to the image of the first."""
    shape = (121, 121)
    psf = GaussianPSF(fwhm=3.0)
    reference = dict(
        flux=100.0, re=10.0, n=4.0, q=1.0, theta=0.0, x0=60.0, y0=60.0,
    )
    target = make_single_sersic_psf_model(**reference, psf=psf).render(shape)

    def residual(values):
        flux, re = values
        model = make_single_sersic_psf_model(
            flux=flux, re=re, n=3.0, q=1.0, theta=0.0, x0=60.0, y0=60.0, psf=psf,
        )
        return (model.render(shape) - target).ravel()

    fit = least_squares(residual, x0=(90.0, 9.0), x_scale=(10.0, 1.0))
    flux, re = fit.x
    matched = make_single_sersic_psf_model(
        flux=flux, re=re, n=3.0, q=1.0, theta=0.0, x0=60.0, y0=60.0, psf=psf,
    )
    rmax = 55
    r, reference_profile = radial_profile(target, 60.0, 60.0, rmax)
    _, matched_profile = radial_profile(matched.render(shape), 60.0, 60.0, rmax)

    fig, axes = panels(1, 2)
    left, right = axes[0]
    first, second = theme.series[0], theme.series[1]
    left.plot(r, reference_profile, color=first,
              label=r"$n = 4$, $R_e = 10$ px, $F = 100$")
    left.plot(
        r, matched_profile, color=second, linestyle="--",
        label=rf"$n = 3$, $R_e = {re:.1f}$ px, $F = {flux:.1f}$",
    )
    left.set_yscale("log")
    left.set_xlim(0, rmax)
    left.set_xlabel(r"$R$ [pix]")
    left.set_ylabel("Mean intensity")
    left.set_title("Two profiles after the PSF")
    left.legend(loc="upper right")
    difference = 100.0 * (matched_profile - reference_profile) / reference_profile
    right.plot(r, difference, color=theme.ink)
    reference_line(right, theme, y=0.0)
    right.set_xlim(0, rmax)
    right.set_xlabel(r"$R$ [pix]")
    right.set_ylabel("Difference [%]")
    right.set_title(r"$n = 3$ relative to $n = 4$")
    for ax in (left, right):
        canonical_axes(ax)
    return fig


def bulge_disk_tradeoff(theme: Theme) -> plt.Figure:
    """Two bulge and disc decompositions of almost the same image."""
    shape = (121, 121)
    psf = GaussianPSF(fwhm=3.0)
    common = dict(
        q_bulge=1.0, theta_bulge=0.0, q_disk=1.0, theta_disk=0.0,
        x0=60.0, y0=60.0, psf=psf,
    )
    reference = dict(
        flux_bulge=30.0, re_bulge=3.0, n_bulge=4.0, flux_disk=70.0, re_disk=16.0,
    )
    target = make_bulge_disk_psf_model(**reference, **common).render(shape)
    total_flux = reference["flux_bulge"] + reference["flux_disk"]
    bulge_fraction = 0.45

    def model_for(values):
        re_bulge, n_bulge, re_disk = values
        return make_bulge_disk_psf_model(
            flux_bulge=bulge_fraction * total_flux, re_bulge=re_bulge,
            n_bulge=n_bulge, flux_disk=(1 - bulge_fraction) * total_flux,
            re_disk=re_disk, **common,
        )

    fit = least_squares(
        lambda values: (model_for(values).render(shape) - target).ravel(),
        x0=(4.0, 3.0, 18.0),
        bounds=((0.5, 0.6, 5.0), (30.0, 8.0, 60.0)),
    )
    alternative = model_for(fit.x)
    first = make_bulge_disk_psf_model(**reference, **common)
    rmax = 55
    fig = plt.figure(figsize=canonical_figsize(2, 2))
    grid = fig.add_gridspec(2, 2, wspace=0.32, hspace=0.38)
    bulge_colour = theme.series[ROLE_SLOT["bulge"]]
    disk_colour = theme.series[ROLE_SLOT["disk"]]
    for column, (model, title) in enumerate(
        (
            (first, r"$B/T = 0.30$"),
            (alternative, rf"$B/T = {bulge_fraction:.2f}$"),
        )
    ):
        ax = fig.add_subplot(grid[0, column])
        parts = model.render_components(shape)
        for name, colour, label in (
            ("bulge", bulge_colour, "Bulge"),
            ("disk", disk_colour, "Disc"),
        ):
            r, profile = radial_profile(parts[name], 60.0, 60.0, rmax)
            ax.plot(r, profile, color=colour, label=label)
        r, profile = radial_profile(parts["bulge"] + parts["disk"], 60.0, 60.0, rmax)
        ax.plot(r, profile, color=theme.ink, linewidth=LINE + 1, label="Total")
        ax.set_yscale("log")
        ax.set_ylim(1e-3, 30)
        ax.set_xlim(0, rmax)
        ax.set_xlabel(r"$R$ [pix]")
        ax.set_ylabel("Mean intensity")
        ax.set_title(title)
        if column == 0:
            ax.legend(loc="upper right")
        canonical_axes(ax)
    ax = fig.add_subplot(grid[1, :])
    r, reference_profile = radial_profile(target, 60.0, 60.0, rmax)
    _, alternative_profile = radial_profile(alternative.render(shape), 60.0, 60.0, rmax)
    ax.plot(
        r, 100.0 * (alternative_profile - reference_profile) / reference_profile,
        color=theme.ink,
    )
    reference_line(ax, theme, y=0.0)
    ax.set_xlim(0, rmax)
    ax.set_xlabel(r"$R$ [pix]")
    ax.set_ylabel("Difference of totals [%]")
    ax.set_title(r"$B/T = 0.45$ relative to $B/T = 0.30$")
    canonical_axes(ax)
    return fig


# ---------------------------------------------------------------------------
# The morphology families beyond the four Sérsic-based models
# ---------------------------------------------------------------------------


def boxy_sersic_isophotes(theme: Theme) -> plt.Figure:
    """Discy, elliptical and boxy isophotes of one Sérsic profile."""
    shape = (81, 81)
    parameters = dict(
        flux=100.0, re=10.0, n=2.0, q=0.7, theta=np.deg2rad(30.0),
        x0=40.0, y0=40.0,
    )
    shapes = ((-0.6, "discy"), (0.0, "ellipses"), (0.6, "boxy"))
    fig, axes = panels(2, 2)
    yy, xx = np.indices(shape, dtype=float)
    major, minor = major_minor_offsets(
        xx, yy, x0=parameters["x0"], y0=parameters["y0"],
        theta=parameters["theta"],
    )
    images = [
        GeneralizedSersicComponent(c0=c0, **parameters).render(shape)
        for c0, _ in shapes
    ]
    peak = max(float(image.max()) for image in images)
    norm = LogNorm(vmin=1e-3 * peak, vmax=peak)
    overlay = axes[1, 1]
    for ax, image, (c0, label), colour in zip(
        axes.flat[:3], images, shapes, theme.series, strict=False
    ):
        show_image(ax, image, theme, norm)
        radius = generalized_radius(major, minor, q=parameters["q"], c0=c0)
        ax.contour(
            radius, levels=(5.0, 10.0, 20.0), colors=theme.series[1],
            linewidths=THIN,
        )
        ax.set_title(rf"$c_0 = {signed(c0)}$, {label}")
        overlay.contour(radius, levels=(20.0,), colors=colour, linewidths=LINE)
        overlay.plot([], [], color=colour, label=rf"$c_0 = {signed(c0)}$")
    overlay.set_aspect("equal")
    overlay.set_xlim(0, shape[1] - 1)
    overlay.set_ylim(0, shape[0] - 1)
    overlay.set_xlabel(r"$x$ [pix]")
    overlay.set_ylabel(r"$y$ [pix]")
    overlay.set_title("One isophote of each")
    overlay.legend(loc="upper left")
    canonical_axes(overlay)
    return fig


def boxy_sersic_profiles(theme: Theme) -> plt.Figure:
    """The light along the axes and along the diagonal of the isophotes."""
    parameters = dict(
        flux=100.0, re=10.0, n=2.0, q=0.7, theta=0.0, x0=0.0, y0=0.0,
    )
    distance = np.linspace(0.0, 30.0, 300)
    q = parameters["q"]
    # Along the major axis the generalized radius is |x'| for every c0; along
    # the direction y' = q x' it depends on c0.
    diagonal = np.array((1.0, q)) / np.hypot(1.0, q)
    fig, axes = panels(1, 2)
    left, right = axes[0]

    def scaled(component, x, y):
        # Each profile in units of its own central intensity, which at a
        # fixed flux falls as the area factor K(c) grows.
        return component.intensity(x, y) / np.exp(component.log_central_intensity)

    ellipse = GeneralizedSersicComponent(c0=0.0, **parameters)
    reference = scaled(ellipse, diagonal[0] * distance, diagonal[1] * distance)
    for (c0, label), colour in zip(
        ((-0.6, "discy"), (0.0, "ellipses"), (0.6, "boxy")),
        theme.series,
        strict=False,
    ):
        component = GeneralizedSersicComponent(c0=c0, **parameters)
        along = scaled(component, diagonal[0] * distance, diagonal[1] * distance)
        name = rf"$c_0 = {signed(c0)}$, {label}"
        left.plot(distance, along, color=colour, label=name)
        right.plot(distance, along / reference, color=colour, label=name)
    major = scaled(ellipse, distance, 0.0 * distance)
    left.plot(distance, major, color=theme.ink, linestyle="--", linewidth=THIN,
              label=r"Major axis, every $c_0$")
    left.set_yscale("log")
    left.set_ylim(1e-4, 1.5)
    left.set_ylabel(r"$I / I_0$")
    left.set_title(r"Along the diagonal $y' = q\,x'$")
    left.legend(loc="lower left")
    reference_line(right, theme, y=1.0)
    right.set_ylabel("Ratio to ellipses")
    right.set_title(r"Diagonal, relative to $c_0 = 0$")
    right.legend(loc="upper left")
    for ax in (left, right):
        ax.set_xlim(0, 30)
        ax.set_xlabel("Distance from the centre [pix]")
        canonical_axes(ax)
    return fig


def core_sersic_profiles(theme: Theme) -> plt.Figure:
    """The depleted core against the inward extrapolation of the Sérsic body."""
    base = dict(
        flux=100.0, re=20.0, n=4.0, rb=2.0, gamma=0.3, alpha=2.0, q=1.0,
        theta=0.0, x0=0.0, y0=0.0,
    )
    radius = np.logspace(-2.0, 1.6, 400)
    fig, axes = panels(1, 2)
    left, right = axes[0]

    def shape(**changes):
        component = CoreSersicComponent(**{**base, **changes})
        profile = component.surface_brightness(radius) / np.exp(
            component.log_intensity_scale
        )
        sersic = np.exp(
            -component.bn * (radius / component.re) ** (1.0 / component.n)
        )
        return profile, sersic

    for ax, key, symbol, values, title in (
        (left, "gamma", r"\gamma", (0.0, 0.3, 0.9),
         r"Inner slope $\gamma$ ($\alpha = 2$)"),
        (right, "alpha", r"\alpha", (0.5, 1.0, 2.0),
         r"Transition $\alpha$ ($\gamma = 0.3$)"),
    ):
        for value, colour in zip(values, theme.series, strict=False):
            profile, sersic = shape(**{key: value})
            ax.plot(radius / base["rb"], profile, color=colour,
                    label=rf"${symbol} = {value:g}$")
        ax.plot(radius / base["rb"], sersic, color=theme.ink, linestyle="--",
                linewidth=THIN, label=cm("Sérsic body"))
        reference_line(ax, theme, x=1.0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$R / R_b$")
        ax.set_title(title)
        ax.legend(loc="lower left")
        canonical_axes(ax)
    left.set_ylabel(r"$I(R) / I'$")
    return fig


def core_sersic_deficit(theme: Theme) -> plt.Figure:
    """A core-Sérsic galaxy, its Sérsic body and the light the core lacks."""
    shape = (101, 101)
    psf = GaussianPSF(fwhm=2.5)
    parameters = dict(
        re=20.0, n=4.0, q=0.8, theta=np.deg2rad(30.0), x0=50.0, y0=50.0,
    )
    core = dict(flux=100.0, rb=4.0, gamma=0.1, alpha=2.0)
    component = CoreSersicComponent(**core, **parameters)
    # Beyond the break the profile is the Sérsic law with central intensity
    # I', whose flux over the plane is 2 pi q I' re^2 n b^-2n Gamma(2n).
    body_flux = float(
        2.0 * np.pi * parameters["q"] * np.exp(component.log_intensity_scale)
        * parameters["re"] ** 2 * parameters["n"]
        * component.bn ** (-2.0 * parameters["n"])
        * np.exp(gammaln(2.0 * parameters["n"]))
    )
    observed = make_core_sersic_psf_model(**core, **parameters, psf=psf).render(
        shape
    )
    body = make_single_sersic_psf_model(
        flux=body_flux, **parameters, psf=psf
    ).render(shape)
    deficit = body - observed
    norm = LogNorm(vmin=1e-4 * float(body.max()), vmax=float(body.max()))
    fig, axes = panels(2, 2)
    lacking = 100.0 * (body_flux - core["flux"]) / body_flux
    for ax, (image, label) in zip(
        axes.flat[:3],
        (
            (observed, cm("Core-Sérsic")),
            (body, cm("Sérsic body")),
            (deficit, rf"Deficit, ${lacking:.1f}\,\%$"),
        ),
        strict=True,
    ):
        show_image(ax, image, theme, norm)
        ax.set_title(label)
    ax = axes[1, 1]
    rmax = 30
    for image, colour, style, label in (
        (observed, theme.series[0], "-", cm("Core-Sérsic")),
        (body, theme.ink, "--", cm("Sérsic body")),
        (deficit, theme.series[1], "-", "Deficit"),
    ):
        r, profile = radial_profile(image, 50.0, 50.0, rmax)
        ax.plot(r, profile, color=colour, linestyle=style, label=label)
    r, profile = radial_profile(body, 50.0, 50.0, rmax)
    reference_line(ax, theme, x=core["rb"])
    ax.set_yscale("log")
    ax.set_ylim(1e-4 * float(profile.max()), 2.0 * float(profile.max()))
    ax.set_xlim(0, rmax)
    ax.set_xlabel(r"$R$ [pix]")
    ax.set_ylabel("Mean intensity")
    ax.set_title("Radial profiles")
    ax.legend(loc="upper right")
    canonical_axes(ax)
    return fig


def broken_disk_profiles(theme: Theme) -> plt.Figure:
    """The three types of disc profile and the width of the break."""
    radius = np.linspace(0.0, 40.0, 500)
    base = dict(
        flux=100.0, h_in=6.0, rb=15.0, wb=1.0, q=1.0, theta=0.0, x0=0.0,
        y0=0.0,
    )
    fig, axes = panels(1, 2)
    left, right = axes[0]

    def shape(**changes):
        disk = BrokenExponentialComponent(**{**base, **changes})
        return disk.surface_brightness(radius) / np.exp(
            disk.log_central_intensity
        )

    for (label, h_out), colour in zip(
        (("Type I", 6.0), ("Type II", 3.0), ("Type III", 12.0)),
        theme.series,
        strict=False,
    ):
        left.plot(radius, shape(h_out=h_out), color=colour,
                  label=rf"{label}, $h_{{\rm out}} = {h_out:g}$ px")
    for width, colour in zip((0.5, 2.0, 5.0), theme.series, strict=False):
        right.plot(radius, shape(h_out=3.0, wb=width), color=colour,
                   label=rf"$w_b = {width:g}$ px")
    for ax, title in (
        (left, r"Disc types, $h_{\rm in} = 6$ px"),
        (right, "Width of the break, type II"),
    ):
        reference_line(ax, theme, x=base["rb"])
        ax.set_yscale("log")
        ax.set_ylim(1e-5, 1.5)
        ax.set_xlim(0, 40)
        ax.set_xlabel(r"$r$ [pix]")
        ax.set_title(title)
        ax.legend(loc="lower left")
        canonical_axes(ax)
    left.set_ylabel(r"$I(r) / I_0$")
    return fig


def bulge_broken_disk_decomposition(theme: Theme) -> plt.Figure:
    """A Sérsic bulge and a type II disc."""
    model = make_bulge_broken_disk_psf_model(
        flux_bulge=30.0, re_bulge=3.0, n_bulge=3.5, q_bulge=0.9,
        theta_bulge=np.deg2rad(40.0),
        flux_disk=70.0, h_in_disk=10.0, h_out_disk=4.0, rb_disk=22.0,
        wb_disk=1.5, q_disk=0.6, theta_disk=np.deg2rad(40.0),
        x0=50.0, y0=50.0, psf=GaussianPSF(fwhm=2.5),
    )
    return decomposition(
        theme, model, ("bulge", "disk"), (r"Bulge, $n = 3.5$", "Type II disc"),
        r"Total, $B/T = 0.3$", 48,
    )


def ferrers_bar_profiles(theme: Theme) -> plt.Figure:
    """The edge of a Ferrers bar and the shape of its isophotes."""
    fig = plt.figure(figsize=canonical_figsize(2, 2))
    grid = fig.add_gridspec(2, 2, wspace=0.32, hspace=0.38)

    shape = (81, 81)
    parameters = dict(
        flux=100.0, a=30.0, alpha=2.0, q=0.35, theta=np.deg2rad(30.0),
        x0=40.0, y0=40.0,
    )
    yy, xx = np.indices(shape, dtype=float)
    major, minor = major_minor_offsets(
        xx, yy, x0=parameters["x0"], y0=parameters["y0"],
        theta=parameters["theta"],
    )
    images = [
        FerrersBarComponent(c0=c0, **parameters).render(shape)
        for c0 in (0.0, 2.5)
    ]
    peak = max(float(image.max()) for image in images)
    norm = LogNorm(vmin=1e-3 * peak, vmax=peak)
    for column, (image, c0) in enumerate(zip(images, (0.0, 2.5), strict=True)):
        ax = fig.add_subplot(grid[0, column])
        show_image(ax, image, theme, norm)
        radius = generalized_radius(major, minor, q=parameters["q"], c0=c0)
        ax.contour(
            radius, levels=(10.0, 20.0, 29.0), colors=theme.series[2],
            linewidths=THIN,
        )
        label = "ellipses" if c0 == 0.0 else "boxy"
        ax.set_title(rf"$c_0 = {c0:g}$, {label}")

    ax = fig.add_subplot(grid[1, :])
    radius = np.linspace(0.0, 1.15, 600)
    for alpha, colour in zip((0.5, 1.0, 2.0, 5.0), theme.series, strict=False):
        bar = FerrersBarComponent(
            flux=1.0, a=1.0, alpha=alpha, c0=0.0, q=1.0, theta=0.0, x0=0.0,
            y0=0.0,
        )
        profile = bar.intensity(radius, np.zeros_like(radius))
        ax.plot(
            radius, profile / np.exp(bar.log_central_intensity), color=colour,
            label=rf"$\alpha = {alpha:g}$",
        )
    ax.set_xlim(0, 1.15)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel(r"$m / a$")
    ax.set_ylabel(r"$I(m) / I_0$")
    ax.set_title("Edge of the bar")
    ax.legend(loc="lower left")
    canonical_axes(ax)
    return fig


def bulge_disk_bar_decomposition(theme: Theme) -> plt.Figure:
    """A Sérsic bulge, an exponential disc and a Ferrers bar."""
    model = make_bulge_disk_bar_psf_model(
        flux_bulge=20.0, re_bulge=3.0, n_bulge=3.0, q_bulge=0.9,
        theta_bulge=np.deg2rad(40.0),
        flux_disk=55.0, re_disk=18.0, q_disk=0.8, theta_disk=np.deg2rad(40.0),
        flux_bar=25.0, a_bar=20.0, alpha_bar=2.0, c0_bar=1.5, q_bar=0.3,
        theta_bar=np.deg2rad(100.0),
        x0=50.0, y0=50.0, psf=GaussianPSF(fwhm=2.5),
    )
    return decomposition(
        theme, model, ("bulge", "disk", "bar"),
        (r"Bulge, $n = 3$", "Disc", "Bar"), "Total", 48,
    )


def gaussian_ring_profiles(theme: Theme) -> plt.Figure:
    """Gaussian rings of three widths and the same flux, and an inclined ring."""
    fig, axes = panels(1, 2)
    ax = axes[0, 0]
    radius = np.linspace(0.0, 45.0, 600)
    for sigma, colour in zip((1.0, 3.0, 6.0), theme.series, strict=False):
        ring = GaussianRingComponent(
            flux=100.0, r=20.0, sigma=sigma, q=1.0, theta=0.0, x0=0.0, y0=0.0,
        )
        ax.plot(
            radius, ring.intensity(radius, np.zeros_like(radius)), color=colour,
            label=rf"$\sigma = {sigma:g}$ px",
        )
    ax.set_xlim(0, 45)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(r"$r$ [pix]")
    ax.set_ylabel("Intensity")
    ax.set_title(r"Rings at $r = 20$ px, same flux")
    ax.legend(loc="upper right")
    canonical_axes(ax)

    shape = (81, 81)
    image = GaussianRingComponent(
        flux=100.0, r=26.0, sigma=2.5, q=0.6, theta=np.deg2rad(30.0),
        x0=40.0, y0=40.0,
    ).render(shape)
    ax = axes[0, 1]
    show_image(
        ax, image, theme, LogNorm(vmin=1e-3 * image.max(), vmax=image.max())
    )
    ax.set_title(r"$r = 26$ px, $\sigma = 2.5$ px, $q = 0.6$")
    return fig


def bulge_disk_ring_decomposition(theme: Theme) -> plt.Figure:
    """A Sérsic bulge, an exponential disc and a Gaussian ring."""
    model = make_bulge_disk_ring_psf_model(
        flux_bulge=25.0, re_bulge=3.0, n_bulge=3.0, q_bulge=0.9,
        theta_bulge=np.deg2rad(40.0),
        flux_disk=60.0, re_disk=16.0, q_disk=0.8, theta_disk=np.deg2rad(40.0),
        flux_ring=15.0, r_ring=26.0, sigma_ring=2.5, q_ring=0.8,
        theta_ring=np.deg2rad(40.0),
        x0=50.0, y0=50.0, psf=GaussianPSF(fwhm=2.5),
    )
    return decomposition(
        theme, model, ("bulge", "disk", "ring"),
        (r"Bulge, $n = 3$", "Disc", "Ring"), "Total", 48,
    )


def spiral_arms(theme: Theme) -> plt.Figure:
    """Arms that draw a Z, an S, and tighter arms, with their crests."""
    shape = (101, 101)
    base = dict(
        flux=100.0, re=20.0, q=1.0, theta=0.0, amplitude=0.7, phase=0.4,
        r_in=4.0, x0=50.0, y0=50.0,
    )
    cases = (
        (2.0, r"$w = 2$: a Z"),
        (-2.0, r"$w = -2$: an S"),
        (6.0, r"$w = 6$: $\psi = 9.5^\circ$"),
    )
    discs = [LogSpiralDiskComponent(winding=w, **base) for w, _ in cases]
    images = [disc.render(shape) for disc in discs]
    peak = max(float(image.max()) for image in images)
    norm = LogNorm(vmin=3e-3 * peak, vmax=peak)
    fig, axes = panels(2, 2)
    radius = np.linspace(base["r_in"], 48.0, 400)
    for ax, disc, image, (winding, label) in zip(
        axes.flat[:3], discs, images, cases, strict=True
    ):
        show_image(ax, image, theme, norm)
        for arm in (0.0, np.pi):
            azimuth = disc.phase + winding * np.log(radius / disc.re) + arm
            ax.plot(
                disc.x0 + radius * np.cos(azimuth),
                disc.y0 + radius * np.sin(azimuth),
                color=theme.series[2], linewidth=THIN,
            )
        ax.set_xlim(-0.5, shape[1] - 0.5)
        ax.set_ylim(-0.5, shape[0] - 0.5)
        ax.set_title(label)
    # A logarithmic spiral of winding w crosses every circle at the pitch
    # angle psi = arctan(1 / |w|).
    pitch = axes[1, 1]
    winding = np.linspace(0.5, 10.0, 400)
    for sign in (-1.0, 1.0):
        pitch.plot(sign * winding, np.rad2deg(np.arctan(1.0 / winding)),
                   color=theme.ink)
    for (value, _), colour in zip(cases, theme.series, strict=False):
        pitch.plot(
            value, np.rad2deg(np.arctan(1.0 / abs(value))), "o",
            color=colour, markersize=MARKER, markeredgecolor=theme.surface,
            markeredgewidth=3, label=rf"$w = {value:g}$",
        )
    pitch.set_xlim(-10, 10)
    pitch.set_ylim(0, 65)
    pitch.set_xlabel(r"Winding $w$")
    pitch.set_ylabel(r"Pitch angle $\psi$ [deg]")
    pitch.set_title(r"$\psi = \arctan(1/|w|)$")
    pitch.legend(loc="upper right")
    canonical_axes(pitch)
    return fig


def bulge_disk_spiral_decomposition(theme: Theme) -> plt.Figure:
    """A Sérsic bulge and an exponential disc with two arms."""
    model = make_bulge_disk_spiral_psf_model(
        flux_bulge=25.0, re_bulge=3.0, n_bulge=3.0, q_bulge=0.9,
        theta_bulge=np.deg2rad(40.0),
        flux_disk=75.0, re_disk=16.0, q_disk=0.8, theta_disk=np.deg2rad(40.0),
        amplitude_spiral=0.6, winding_spiral=2.5, phase_spiral=0.4,
        r_in_spiral=4.0,
        x0=50.0, y0=50.0, psf=GaussianPSF(fwhm=2.5),
    )
    return decomposition(
        theme, model, ("bulge", "disk"), (r"Bulge, $n = 3$", "Disc with arms"),
        r"Total, $B/T = 0.25$", 48,
    )


FIGURES: dict[str, Callable[[Theme], plt.Figure]] = {
    "forward_model": forward_model,
    "model_composition": model_composition,
    "geometry": geometry,
    "sersic_profiles": sersic_profiles,
    "point_source_placement": point_source_placement,
    "single_sersic_parameters": single_sersic_parameters,
    "double_sersic_decomposition": double_sersic_decomposition,
    "sersic_point_source_decomposition": sersic_point_source_decomposition,
    "bulge_disk_decomposition": bulge_disk_decomposition,
    "gaussian_psf": gaussian_psf,
    "effective_psf_window": effective_psf_window,
    "background_models": background_models,
    "sky_region": sky_region,
    "noise_variance": noise_variance,
    "correlated_noise": correlated_noise,
    "sersic_degeneracy": sersic_degeneracy,
    "bulge_disk_tradeoff": bulge_disk_tradeoff,
    "boxy_sersic_isophotes": boxy_sersic_isophotes,
    "boxy_sersic_profiles": boxy_sersic_profiles,
    "core_sersic_profiles": core_sersic_profiles,
    "core_sersic_deficit": core_sersic_deficit,
    "broken_disk_profiles": broken_disk_profiles,
    "bulge_broken_disk_decomposition": bulge_broken_disk_decomposition,
    "ferrers_bar_profiles": ferrers_bar_profiles,
    "bulge_disk_bar_decomposition": bulge_disk_bar_decomposition,
    "gaussian_ring_profiles": gaussian_ring_profiles,
    "bulge_disk_ring_decomposition": bulge_disk_ring_decomposition,
    "spiral_arms": spiral_arms,
    "bulge_disk_spiral_decomposition": bulge_disk_spiral_decomposition,
}


# ---------------------------------------------------------------------------
# Theory
# ---------------------------------------------------------------------------


def likelihood_and_occam(theme: Theme) -> plt.Figure:
    """A correlated likelihood, and the penalty of the evidence on a wide prior.

    Left: the likelihood of the radius and the index of a simulated Sérsic
    galaxy, computed with BayMorph, with the flux at its best value for each
    pair. Right: one likelihood under a narrow and a wide prior.
    """
    shape = (61, 61)
    psf = GaussianPSF(fwhm=2.5)
    geometry_ = dict(q=0.8, theta=np.deg2rad(30.0), x0=30.0, y0=30.0)
    truth = dict(flux=500.0, re=6.0, n=2.5)
    noise = 0.5
    data = make_single_sersic_psf_model(**truth, **geometry_, psf=psf).render(
        shape
    ) + np.random.default_rng(13).normal(0.0, noise, shape)
    likelihood = GaussianImageLikelihood(data, noise)
    radii = np.linspace(4.5, 8.0, 36)
    indices = np.linspace(1.8, 3.4, 36)
    log_like = np.empty((indices.size, radii.size))
    for row, n in enumerate(indices):
        for column, re in enumerate(radii):
            unit = make_single_sersic_psf_model(
                flux=1.0, re=re, n=n, **geometry_, psf=psf
            ).render(shape)
            flux = float(np.sum(unit * data) / np.sum(unit * unit))
            log_like[row, column] = likelihood(flux * unit)
    delta = log_like.max() - log_like
    fig, axes = panels(1, 2)
    ax = axes[0, 0]
    # Two-parameter confidence levels of 68.3, 95.4 and 99.7 percent.
    levels = (0.0, 1.15, 3.09, 5.92)
    ax.contourf(radii, indices, delta, levels=levels,
                colors=(theme.image_ramp[4], theme.image_ramp[3],
                        theme.image_ramp[2]))
    ax.contour(radii, indices, delta, levels=levels[1:], colors=theme.ink,
               linewidths=THIN)
    ax.plot(truth["re"], truth["n"], "*", color=theme.series[1],
            markersize=MARKER * 1.6, markeredgecolor=theme.surface,
            markeredgewidth=2, label="truth")
    ax.set_xlabel(r"$R_e$ [pix]")
    ax.set_ylabel(r"$n$")
    ax.set_title(r"The likelihood of $R_e$ and $n$")
    ax.legend(loc="upper left")
    canonical_axes(ax)

    ax = axes[0, 1]
    theta = np.linspace(-6.0, 6.0, 4001)
    like = np.exp(-0.5 * ((theta - 0.4) / 0.25) ** 2)
    ax.plot(theta, like, color=theme.ink, linestyle="--", linewidth=THIN,
            label=r"likelihood $L$")
    for (lower, upper), colour, label in (
        ((-1.0, 2.0), theme.series[0], "narrow"),
        ((-5.0, 5.0), theme.series[1], "wide"),
    ):
        prior = np.where((theta >= lower) & (theta <= upper),
                         1.0 / (upper - lower), 0.0)
        integrand = like * prior
        evidence = float(np.trapezoid(integrand, theta))
        ax.fill_between(theta, integrand, color=colour, alpha=0.35,
                        linewidth=0)
        ax.plot(theta, integrand, color=colour,
                label=rf"$L\,\pi$, {label} prior: $Z = {evidence:.3f}$")
    ax.set_xlim(-6, 6)
    ax.set_ylim(0, 1.25)
    ax.set_xlabel(r"Parameter $\theta$")
    ax.set_ylabel("Density")
    ax.set_title("The evidence penalizes a wide prior")
    ax.legend(loc="upper left")
    canonical_axes(ax)
    return fig


def nested_sampling(theme: Theme) -> plt.Figure:
    """A small nested sampling run on a two-dimensional Gaussian likelihood.

    The prior is uniform on a square of side 10 and the likelihood a circular
    Gaussian of width 0.5, so the evidence is known analytically. Each new
    point is drawn uniformly from the region of higher likelihood, a disc,
    as an ideal sampler would; real samplers have to construct that region.
    """
    rng = np.random.default_rng(5)
    sigma, half, n_live, iterations = 0.5, 5.0, 300, 3300

    def log_like(points):
        return -0.5 * np.sum(points**2, axis=-1) / sigma**2

    live = rng.uniform(-half, half, (n_live, 2))
    live_log_like = log_like(live)
    dead, dead_log_like = [], []
    for _ in range(iterations):
        worst = int(np.argmin(live_log_like))
        threshold = live_log_like[worst]
        dead.append(live[worst].copy())
        dead_log_like.append(threshold)
        radius = sigma * np.sqrt(-2.0 * threshold)
        while True:
            if radius >= half:
                candidate = rng.uniform(-half, half, 2)
            else:
                angle = rng.uniform(0.0, 2.0 * np.pi)
                distance = radius * np.sqrt(rng.uniform())
                candidate = distance * np.array((np.cos(angle), np.sin(angle)))
            if log_like(candidate) > threshold:
                break
        live[worst] = candidate
        live_log_like[worst] = log_like(candidate)
    dead = np.array(dead)
    dead_log_like = np.array(dead_log_like)
    steps = np.arange(1, iterations + 1)
    log_volume = -steps / n_live
    volume = np.exp(log_volume)
    shells = np.exp(np.concatenate(([0.0], log_volume[:-1]))) - volume
    weights = np.exp(dead_log_like) * shells
    running = np.cumsum(weights)
    remainder = volume[-1] * float(np.mean(np.exp(live_log_like)))
    evidence = running[-1] + remainder
    analytic = np.log(2.0 * np.pi * sigma**2 / (2.0 * half) ** 2)
    posterior = weights / evidence
    information = float(np.sum(posterior * (dead_log_like - np.log(evidence))))
    error = np.sqrt(information / n_live)

    fig, axes = panels(2, 2)
    ax = axes[0, 0]
    order = np.argsort(steps)
    ax.scatter(dead[order, 0], dead[order, 1], c=steps[order], s=25,
               cmap=image_colormap(theme), linewidths=0)
    ax.scatter(live[:, 0], live[:, 1], s=60, color=theme.series[1],
               linewidths=0, label="live points at the end")
    ax.set_xlim(-half, half)
    ax.set_ylim(-half, half)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$\theta_1$")
    ax.set_ylabel(r"$\theta_2$")
    ax.set_title("Discarded points, by iteration")
    ax.legend(loc="upper left")
    canonical_axes(ax)

    ax = axes[0, 1]
    ax.plot(log_volume, dead_log_like, color=theme.series[0])
    ax.set_xlabel(r"$\ln X$, prior volume")
    ax.set_ylabel(r"$\ln L$")
    ax.set_title("Likelihood against prior volume")
    ax.set_xlim(log_volume[-1], 0)
    ax.set_ylim(-25.0, 1.0)
    canonical_axes(ax)

    ax = axes[1, 0]
    ax.plot(log_volume, posterior / posterior.max(), color=theme.series[2])
    ax.set_xlabel(r"$\ln X$, prior volume")
    ax.set_ylabel(r"Weight $L_i\,\Delta X_i$ [arbitrary units]")
    ax.set_title("Where the posterior mass lies")
    ax.set_xlim(log_volume[-1], 0)
    ax.set_ylim(0, 1.1)
    canonical_axes(ax)

    ax = axes[1, 1]
    ax.plot(steps, np.log(running), color=theme.series[0],
            label=r"running $\ln Z$")
    ax.axhspan(np.log(evidence) - error, np.log(evidence) + error,
               color=theme.series[0], alpha=0.2, linewidth=0)
    reference_line(ax, theme, y=analytic)
    ax.text(iterations * 0.98, analytic - 0.35, r"exact $\ln Z$",
            color=theme.ink, fontsize=NOTE_SIZE, ha="right", va="top")
    ax.set_xlim(0, iterations)
    ax.set_ylim(analytic - 4.0, analytic + 1.0)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$\ln Z$")
    ax.set_title("The evidence as the run proceeds")
    ax.legend(loc="lower right")
    canonical_axes(ax)
    return fig


def hierarchical_coupling(theme: Theme) -> plt.Figure:
    """The prior of a hierarchical parameter in two images.

    The samples are drawn with the coupling of BayMorph: two standard-normal
    latent values, made equicorrelated with correlation rho and mapped
    through the declared prior, which each image therefore keeps exactly.
    """
    prior = UniformPrior(0.5, 5.0)
    rng = np.random.default_rng(17)
    latents = rng.normal(0.0, 1.0, (1500, 2))
    fig, axes = panels(2, 2)
    for ax, rho, colour in zip(
        axes.flat[:3], (0.0, 0.6, 0.95), theme.series, strict=False
    ):
        values = np.array(
            [physical_prior_copula_values(prior, latent, rho)
             for latent in latents]
        )
        ax.scatter(values[:, 0], values[:, 1], s=30, color=colour,
                   alpha=0.6, linewidths=0)
        ax.set_xlim(0.5, 5.0)
        ax.set_ylim(0.5, 5.0)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$R_e$ in image $g$ [arcsec]")
        ax.set_ylabel(r"$R_e$ in image $r$ [arcsec]")
        ax.set_title(rf"$\rho = {rho:g}$")
        canonical_axes(ax)

    ax = axes[1, 1]
    ratio = np.logspace(0.0, np.log10(4.0), 300)
    separation = np.log(ratio) / np.log(2.0)
    for reference, colour in zip((0.3, 0.6, 0.9), theme.series, strict=False):
        ax.plot(ratio, reference ** (separation**2), color=colour,
                label=rf"$r_\star = {reference:g}$")
    reference_line(ax, theme, x=2.0)
    ax.set_xscale("log")
    ax.set_xlim(1.0, 4.0)
    ax.set_ylim(0, 1.05)
    ax.set_xticks((1.0, 2.0, 3.0, 4.0))
    ax.set_xticklabels(("1", "2", "3", "4"))
    ax.set_xlabel(r"Wavelength ratio $\lambda_2 / \lambda_1$")
    ax.set_ylabel("Correlation of the latent values")
    ax.set_title("Wavelength coupling")
    ax.legend(loc="lower left")
    canonical_axes(ax)
    return fig


THEORY_FIGURES: dict[str, Callable[[Theme], plt.Figure]] = {
    "likelihood_and_occam": likelihood_and_occam,
    "nested_sampling": nested_sampling,
    "hierarchical_coupling": hierarchical_coupling,
}


def main(argv: list[str] | None = None) -> int:
    """Write every figure, or the named ones, in both colour schemes."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("names", nargs="*", help="figures to write; all by default")
    parser.add_argument(
        "--output", type=Path, default=None,
        help="directory for every figure; by default the Models and Theory "
             "figures go to their own directories",
    )
    arguments = parser.parse_args(argv)
    registry = {
        **{name: (function, OUTPUT_DIRECTORY) for name, function in FIGURES.items()},
        **{
            name: (function, THEORY_DIRECTORY)
            for name, function in THEORY_FIGURES.items()
        },
    }
    unknown = sorted(set(arguments.names) - set(registry))
    if unknown:
        parser.error(f"unknown figures: {unknown}; choose from {sorted(registry)}")
    names = arguments.names or list(registry)
    for name in names:
        function, directory = registry[name]
        for theme in THEMES:
            apply_style(theme)
            path = save(function(theme), name, theme, arguments.output or directory)
            print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
