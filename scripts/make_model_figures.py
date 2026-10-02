"""Generate the figures of the Models section of the documentation.

Most figures are computed with BayMorph itself, so they show what the code
does. Two are diagrams of conventions, ``model_composition`` and
``geometry``, and ``sersic_profiles`` evaluates the Sérsic equations with
SciPy. Every figure is written twice to ``docs/models/figures/``:
``<name>-light.png`` and ``<name>-dark.png``, for the two colour schemes of
the documentation site. Run it from the repository root after a change to a
model, a PSF, the background or the effective-PSF rendering::

    python scripts/make_model_figures.py
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
    Rectangle,
)
from scipy.optimize import least_squares  # noqa: E402
from scipy.special import gammainc, gammaincinv  # noqa: E402

from baymorph.fitting import (  # noqa: E402
    make_bulge_disk_psf_model,
    make_double_sersic_psf_model,
    make_sersic_point_source_psf_model,
    make_single_sersic_psf_model,
)
from baymorph.models import PointSourceModel, SersicModel  # noqa: E402
from baymorph.observation.background import (  # noqa: E402
    ConstantBackground,
    TiltedPlaneBackground,
)
from baymorph.observation.point_source import (  # noqa: E402
    PSFPointSourceRenderer,
)
from baymorph.rendering.effective_psf import (  # noqa: E402
    SUBPIXEL_FACTOR,
    WINDOW_HALF,
    WINDOW_RADIUS,
    smooth_window,
)
from baymorph.simulation.psf import GaussianPSF  # noqa: E402

OUTPUT_DIRECTORY = Path("docs") / "models" / "figures"
DPI = 200
WIDTH = 7.5


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

# Compact components (bulge, point source) wear slot 2 and extended ones
# (disc, Sérsic) slot 1 in every figure, so a colour keeps one meaning.
ROLE_SLOT = {"bulge": 1, "point_source": 1, "disk": 0, "sersic": 0}


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------


def apply_style(theme: Theme) -> None:
    """Set the Matplotlib parameters of one colour scheme."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "text.color": theme.ink,
            "axes.labelcolor": theme.secondary,
            "axes.titlecolor": theme.ink,
            "axes.edgecolor": theme.axis,
            "axes.facecolor": theme.surface,
            "axes.linewidth": 0.6,
            "axes.grid": False,
            "grid.color": theme.grid,
            "grid.linewidth": 0.5,
            "grid.linestyle": "-",
            "xtick.color": theme.muted,
            "ytick.color": theme.muted,
            "xtick.labelcolor": theme.secondary,
            "ytick.labelcolor": theme.secondary,
            "figure.facecolor": theme.surface,
            "savefig.facecolor": theme.surface,
            "lines.linewidth": 1.4,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
            "legend.frameon": False,
            "mathtext.fontset": "dejavusans",
        }
    )


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


def plot_axes(ax) -> None:
    """Give a plot recessive chrome: two hairline spines and a grid."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True)
    ax.set_axisbelow(True)


def image_axes(ax, theme: Theme) -> None:
    """Remove ticks from an image panel and keep a hairline frame."""
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(theme.axis)
        spine.set_linewidth(0.6)


def show_image(ax, image, theme: Theme, norm) -> None:
    """Draw an image with its origin at the bottom left, as BayMorph does."""
    ax.imshow(
        np.clip(image, norm.vmin, None) if isinstance(norm, LogNorm) else image,
        origin="lower",
        cmap=image_colormap(theme),
        norm=norm,
        interpolation="nearest",
    )
    image_axes(ax, theme)


def save(fig, name: str, theme: Theme, directory: Path) -> Path:
    """Write one figure and close it."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}-{theme.name}.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.08)
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


def end_label(ax, x, y, text, theme: Theme, dx: float = 4.0) -> None:
    """Write a direct label in text ink just right of a line end."""
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(dx, 0),
        textcoords="offset points",
        va="center",
        ha="left",
        color=theme.secondary,
        fontsize=8,
    )


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
    panels = (
        (intrinsic, "Intrinsic", r"$I(\mathbf{x})$"),
        (convolved, "After the PSF", r"$P \circledast I$"),
        (expected, "Plus background", r"$\mu = P \circledast I + B$"),
        (data, "Observed, with noise", r"$D = \mu + \varepsilon$"),
    )
    fig, axes = plt.subplots(1, 4, figsize=(WIDTH, 2.25))
    fig.subplots_adjust(wspace=0.32)
    for ax, (image, title, formula) in zip(axes, panels, strict=True):
        show_image(ax, image, theme, norm)
        ax.set_title(title, pad=4)
        ax.set_xlabel(formula, color=theme.ink, fontsize=9)
    for left, right in zip(axes[:-1], axes[1:], strict=True):
        start = left.get_position()
        end = right.get_position()
        y = start.y0 + 0.5 * start.height
        fig.add_artist(
            FancyArrowPatch(
                (start.x1 + 0.006, y),
                (end.x0 - 0.006, y),
                transform=fig.transFigure,
                arrowstyle="-|>",
                mutation_scale=9,
                color=theme.muted,
                linewidth=1.0,
            )
        )
    return fig


def model_composition(theme: Theme) -> plt.Figure:
    """Which components each morphology model is made of."""
    fig, ax = plt.subplots(figsize=(WIDTH, 2.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(1.0, 5.75)
    ax.axis("off")

    def box(x, y, w, h, title, detail=None, emphasis=False):
        ax.add_patch(
            FancyBboxPatch(
                (x, y), w, h,
                boxstyle="round,pad=0.02,rounding_size=0.12",
                facecolor=theme.grid if emphasis else theme.surface,
                edgecolor=theme.axis,
                linewidth=0.8,
            )
        )
        if detail is None:
            ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                    color=theme.ink, fontsize=8.5)
            return
        ax.text(x + w / 2, y + h / 2 + 0.15, title, ha="center", va="center",
                color=theme.ink, fontsize=8.5)
        ax.text(x + w / 2, y + h / 2 - 0.18, detail, ha="center", va="center",
                color=theme.secondary, fontsize=7.3)

    def link(start, end):
        ax.add_patch(
            FancyArrowPatch(
                start, end, arrowstyle="-|>", mutation_scale=8,
                color=theme.muted, linewidth=0.9, shrinkA=0, shrinkB=0,
            )
        )

    ax.text(1.3, 5.45, "Components", ha="center", color=theme.secondary)
    ax.text(5.0, 5.45, "Morphology models", ha="center", color=theme.secondary)
    ax.text(8.6, 5.45, "Observation", ha="center", color=theme.secondary)

    box(0.15, 3.85, 2.3, 0.85, "Sérsic profile")
    box(0.15, 2.55, 2.3, 0.85, "Exponential disc", "Sérsic with n = 1")
    box(0.15, 1.25, 2.3, 0.85, "Point source", "unresolved")

    models = (
        ("single_sersic", "one Sérsic", 4.25),
        ("double_sersic", "Sérsic + Sérsic", 3.25),
        ("bulge_disk", "Sérsic bulge + exponential disc", 2.25),
        ("sersic_point_source", "Sérsic + point source", 1.25),
    )
    for name, detail, y in models:
        box(3.55, y, 2.9, 0.78, name, detail, emphasis=True)

    sersic_out = (2.45, 4.27)
    for _, _, y in models:
        link(sersic_out, (3.55, y + 0.39))
    link((2.45, 2.97), (3.55, 2.64))
    link((2.45, 1.67), (3.55, 1.64))

    box(7.35, 2.15, 2.5, 1.65, "convolved with the PSF\n+ residual background\n= expected image")
    for _, _, y in models:
        link((6.45, y + 0.39), (7.35, 2.97))
    return fig


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------


def geometry(theme: Theme) -> plt.Figure:
    """The coordinate and shape conventions of a component."""
    fig, ax = plt.subplots(figsize=(4.6, 3.9))
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 8.5)
    ax.set_aspect("equal")
    for value in np.arange(-0.5, 11.0, 1.0):
        ax.axvline(value, color=theme.grid, linewidth=0.5, zorder=0)
    for value in np.arange(-0.5, 9.0, 1.0):
        ax.axhline(value, color=theme.grid, linewidth=0.5, zorder=0)
    ax.add_patch(
        Rectangle(
            (1.5, 0.5), 1.0, 1.0,
            facecolor=theme.grid, edgecolor=theme.axis, linewidth=0.8,
            zorder=1,
        )
    )
    ax.plot(2.0, 1.0, "o", color=theme.muted, markersize=3, zorder=2)
    ax.annotate(
        "pixel [1, 2], centred at (x, y) = (2, 1)",
        xy=(2.5, 1.0), xytext=(3.4, 0.2),
        color=theme.secondary, fontsize=7.5,
        arrowprops=dict(arrowstyle="-", color=theme.muted, linewidth=0.7),
    )

    x0, y0, re, q, theta = 5.4, 4.6, 3.3, 0.55, np.deg2rad(30.0)
    colour = theme.series[0]
    ax.add_patch(
        Ellipse(
            (x0, y0), 2 * re, 2 * q * re, angle=np.rad2deg(theta),
            fill=False, edgecolor=colour, linewidth=1.6, zorder=3,
        )
    )
    major = (x0 + re * np.cos(theta), y0 + re * np.sin(theta))
    minor = (
        x0 - q * re * np.sin(theta), y0 + q * re * np.cos(theta)
    )
    ax.plot([x0, major[0]], [y0, major[1]], color=colour, linewidth=1.4, zorder=3)
    ax.plot([x0, minor[0]], [y0, minor[1]], color=colour, linewidth=1.4, zorder=3)
    ax.plot([x0, x0 + 2.6], [y0, y0], color=theme.muted, linewidth=0.8, zorder=2)
    ax.add_patch(
        Arc(
            (x0, y0), 2.4, 2.4, theta1=0.0, theta2=np.rad2deg(theta),
            color=theme.ink, linewidth=0.9, zorder=3,
        )
    )
    ax.plot(
        x0, y0, "o", color=colour, markersize=5.5,
        markeredgecolor=theme.surface, markeredgewidth=1.2, zorder=4,
    )
    ax.text(x0 - 0.15, y0 - 0.45, r"$(x_0, y_0)$", ha="right", color=theme.ink)
    ax.text(
        x0 + 0.72 * (major[0] - x0) + 0.25, y0 + 0.72 * (major[1] - y0) - 0.1,
        r"$R_e$", color=theme.ink,
    )
    ax.text(
        0.5 * (x0 + minor[0]) - 0.55, 0.5 * (y0 + minor[1]),
        r"$q\,R_e$", color=theme.ink,
    )
    ax.text(x0 + 1.35, y0 + 0.12, r"$\theta$", color=theme.ink)
    ax.annotate(
        "", xy=(10.3, -0.3), xytext=(-0.3, -0.3),
        arrowprops=dict(arrowstyle="-|>", color=theme.secondary, linewidth=0.9),
    )
    ax.annotate(
        "", xy=(-0.3, 8.3), xytext=(-0.3, -0.3),
        arrowprops=dict(arrowstyle="-|>", color=theme.secondary, linewidth=0.9),
    )
    ax.text(10.3, -0.05, "+x", ha="right", va="bottom", color=theme.secondary)
    ax.text(-0.1, 8.3, "+y", ha="left", va="top", color=theme.secondary)
    ax.set_xticks(range(0, 11, 2))
    ax.set_yticks(range(0, 9, 2))
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    return fig


SERIES_INDICES = (0.5, 1.0, 2.0, 4.0, 6.0)


def sersic_profiles(theme: Theme) -> plt.Figure:
    """Profiles and enclosed light of the Sérsic profile."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(WIDTH, 2.9))
    fig.subplots_adjust(wspace=0.42)
    radius = np.linspace(0.0, 4.0, 400)
    log_radius = np.logspace(-2, 1, 400)
    for index, colour in zip(SERIES_INDICES, theme.series, strict=False):
        bn = float(gammaincinv(2 * index, 0.5))
        profile = np.exp(-bn * (radius ** (1 / index) - 1.0))
        left.plot(radius, profile, color=colour, label=f"n = {index:g}")
        enclosed = gammainc(2 * index, bn * log_radius ** (1 / index))
        right.plot(log_radius, enclosed, color=colour, label=f"n = {index:g}")
    left.set_yscale("log")
    left.set_ylim(1e-6, 3e3)
    left.set_xlim(0, 4)
    left.set_xlabel(r"$R / R_e$")
    left.set_ylabel(r"$I(R) / I_e$")
    left.set_title("Surface brightness", loc="left")
    right.set_xscale("log")
    right.set_xlim(1e-2, 10)
    right.set_ylim(0, 1.02)
    right.axhline(0.5, color=theme.axis, linewidth=0.8, zorder=1)
    right.axvline(1.0, color=theme.axis, linewidth=0.8, zorder=1)
    right.text(
        1.08, 0.04, r"$R = R_e$ holds half the light",
        color=theme.secondary, fontsize=7.5,
    )
    right.set_xlabel(r"$R / R_e$")
    right.set_ylabel(r"$L(<R) / F$")
    right.set_title("Enclosed light", loc="left")
    left.legend(loc="lower left", handlelength=1.4)
    for ax in (left, right):
        plot_axes(ax)
    return fig


def point_source_placement(theme: Theme) -> plt.Figure:
    """A point source is the PSF placed at a sub-pixel position."""
    shape = (11, 11)
    renderer = PSFPointSourceRenderer(GaussianPSF(fwhm=2.5))
    offsets = (0.0, 0.25, 0.5)
    fig, axes = plt.subplots(1, 3, figsize=(5.4, 2.1))
    fig.subplots_adjust(wspace=0.12)
    norm = LogNorm(vmin=1e-4, vmax=0.17)
    for ax, offset in zip(axes, offsets, strict=True):
        source = PointSourceModel(flux=1.0, x0=5.0 + offset, y0=5.0)
        show_image(ax, renderer.render(source, shape), theme, norm)
        ax.plot(
            source.x0, source.y0, "o", color=theme.series[1], markersize=5,
            markeredgecolor=theme.surface, markeredgewidth=1.0,
        )
        ax.set_title(f"offset {offset:g} px", fontsize=8.5)
    return fig


# ---------------------------------------------------------------------------
# Morphology models
# ---------------------------------------------------------------------------


def single_sersic_parameters(theme: Theme) -> plt.Figure:
    """What each parameter of a Sérsic profile does to the image."""
    shape = (61, 61)
    psf = GaussianPSF(fwhm=2.5)
    base = dict(
        flux=100.0, re=7.0, n=2.0, q=0.7, theta=np.deg2rad(30.0),
        x0=30.0, y0=30.0,
    )
    rows = (
        ("n", "n", (0.5, 1.0, 2.5, 5.0), lambda v: f"n = {v:g}"),
        ("re", r"$R_e$", (3.0, 6.0, 10.0, 16.0), lambda v: f"$R_e$ = {v:g} px"),
        ("q", "q", (1.0, 0.8, 0.6, 0.4), lambda v: f"q = {v:g}"),
        (
            "theta", r"$\theta$", (0.0, 45.0, 90.0, 135.0),
            lambda v: f"θ = {v:g}°",
        ),
    )
    fig, axes = plt.subplots(4, 4, figsize=(WIDTH * 0.78, WIDTH * 0.84))
    fig.subplots_adjust(wspace=0.06, hspace=0.32)
    norm = LogNorm(vmin=0.01, vmax=12.0)
    for row, (key, label, values, caption) in enumerate(rows):
        for column, value in enumerate(values):
            parameters = dict(base)
            parameters[key] = np.deg2rad(value) if key == "theta" else value
            image = make_single_sersic_psf_model(**parameters, psf=psf).render(shape)
            ax = axes[row, column]
            show_image(ax, image, theme, norm)
            ax.set_xlabel(caption(value), fontsize=8, labelpad=2)
        axes[row, 0].set_ylabel(label, rotation=0, labelpad=14, fontsize=10, color=theme.ink)
    return fig


def decomposition(theme: Theme, model, names, labels, title_total, rmax):
    """Images of the components and the total, with their radial profiles."""
    shape = (101, 101)
    components = model.render_components(shape)
    total = sum(components[name] for name in names)
    norm = LogNorm(vmin=1e-4 * float(total.max()), vmax=float(total.max()))
    fig = plt.figure(figsize=(WIDTH, 2.2))
    grid = fig.add_gridspec(
        1, 5, width_ratios=(1, 1, 1, 0.42, 1.55), wspace=0.08
    )
    panels = [(components[name], label) for name, label in zip(names, labels, strict=True)]
    panels.append((total, title_total))
    for slot, (image, label) in enumerate(panels):
        ax = fig.add_subplot(grid[0, slot])
        show_image(ax, image, theme, norm)
        ax.set_title(label, fontsize=8.5)
    ax = fig.add_subplot(grid[0, 4])
    for name, label in zip(names, labels, strict=True):
        colour = theme.series[ROLE_SLOT[name]]
        r, profile = radial_profile(components[name], 50.0, 50.0, rmax)
        ax.plot(r, profile, color=colour, label=label)
    r, profile = radial_profile(total, 50.0, 50.0, rmax)
    ax.plot(r, profile, color=theme.ink, linewidth=1.6, label="total")
    ax.set_yscale("log")
    ax.set_ylim(1e-4 * float(profile.max()), 2.0 * float(profile.max()))
    ax.set_xlim(0, rmax)
    ax.set_xlabel("radius (pixels)")
    ax.set_ylabel("mean intensity")
    ax.legend(loc="upper right", handlelength=1.4)
    plot_axes(ax)
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
        theme, model, ("bulge", "disk"), ("bulge component", "disk component"),
        "total", 48,
    )


def sersic_point_source_decomposition(theme: Theme) -> plt.Figure:
    """A Sérsic profile and an unresolved source at the same centre."""
    model = make_sersic_point_source_psf_model(
        flux_sersic=80.0, re=8.0, n=2.0, q=0.7, theta=np.deg2rad(30.0),
        flux_point_source=20.0, x0=50.0, y0=50.0, psf=GaussianPSF(fwhm=2.5),
    )
    return decomposition(
        theme, model, ("sersic", "point_source"), ("Sérsic component", "point source"),
        "total", 40,
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
        theme, model, ("bulge", "disk"), ("bulge, n = 3.5", "exponential disc"),
        "total, B/T = 0.3", 48,
    )


# ---------------------------------------------------------------------------
# PSF and background
# ---------------------------------------------------------------------------


def gaussian_psf(theme: Theme) -> plt.Figure:
    """The Gaussian kernel and its profile."""
    psf = GaussianPSF(fwhm=3.0)
    kernel = psf.kernel()
    fig = plt.figure(figsize=(WIDTH * 0.85, 2.5))
    grid = fig.add_gridspec(1, 2, width_ratios=(1, 1.6), wspace=0.32)
    ax = fig.add_subplot(grid[0, 0])
    show_image(ax, kernel, theme, LogNorm(vmin=1e-6, vmax=float(kernel.max())))
    ax.set_title("kernel, 25 × 25 px", fontsize=8.5)
    ax = fig.add_subplot(grid[0, 1])
    centre = kernel.shape[0] // 2
    yy, xx = np.indices(kernel.shape)
    radius = np.hypot(xx - centre, yy - centre).ravel()
    order = np.argsort(radius)
    ax.plot(
        radius[order], kernel.ravel()[order] / kernel.max(), "o",
        color=theme.series[0], markersize=3.2, markeredgewidth=0,
        label="kernel pixels",
    )
    fine = np.linspace(0, 8, 300)
    ax.plot(
        fine, np.exp(-0.5 * fine**2 / psf.sigma**2), color=theme.ink,
        linewidth=1.2, label="Gaussian",
    )
    ax.axhline(0.5, color=theme.axis, linewidth=0.8)
    ax.annotate(
        "", xy=(0.0, 0.5), xytext=(psf.fwhm / 2, 0.5),
        arrowprops=dict(arrowstyle="<->", color=theme.secondary, linewidth=0.8),
    )
    ax.text(0.15, 0.58, "FWHM / 2", color=theme.secondary, fontsize=7.5)
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("radius (pixels)")
    ax.set_ylabel("normalized value")
    ax.legend(loc="upper right")
    plot_axes(ax)
    return fig


def effective_psf_window(theme: Theme) -> plt.Figure:
    """How the effective-PSF rendering of AD-046 splits a profile."""
    fig = plt.figure(figsize=(WIDTH, 2.8))
    grid = fig.add_gridspec(1, 2, width_ratios=(1.5, 1), wspace=0.28)
    ax = fig.add_subplot(grid[0, 0])
    radius = np.linspace(0.0, 16.0, 800)
    window = smooth_window(radius, np.zeros_like(radius), 0.0, 0.0)
    inner, outer = theme.series[0], theme.series[1]
    ax.plot(radius, window, color=inner, label="w: fine cells, sub-pixel kernels")
    ax.plot(radius, 1.0 - window, color=outer, label="1 − w: pixel centres, native kernel")
    ax.axvline(WINDOW_RADIUS, color=theme.axis, linewidth=0.8)
    ax.axvline(WINDOW_HALF, color=theme.axis, linewidth=0.8)
    ax.text(WINDOW_RADIUS + 0.2, 0.52, "7 px", color=theme.secondary, fontsize=7.5)
    ax.text(WINDOW_HALF - 0.2, 0.52, "cut, 14 px", color=theme.secondary, fontsize=7.5, ha="right")
    ax.set_xlim(0, 16)
    ax.set_ylim(-0.02, 1.25)
    ax.set_xlabel("distance from the component centre (pixels)")
    ax.set_ylabel("weight")
    ax.legend(loc="upper center", ncols=1)
    plot_axes(ax)

    ax = fig.add_subplot(grid[0, 1])
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
            color=colour, linewidth=0.9,
        )
    ax.text(0.0, 0.0, f"cells {SUBPIXEL_FACTOR}×\nfiner", ha="center", va="center",
            color=theme.surface, fontsize=7.5)
    ax.text(0.0, -11.2, "pixel centres", ha="center", va="center",
            color=theme.secondary, fontsize=7.5)
    ax.set_title("window w about the centre", fontsize=8.5)
    image_axes(ax, theme)
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
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH * 0.72, 2.7))
    fig.subplots_adjust(wspace=0.12)
    for ax, image, title in (
        (axes[0], constant, "constant"),
        (axes[1], plane, "plane"),
    ):
        handle = ax.imshow(
            image, origin="lower", cmap=diverging_colormap(theme), norm=norm,
        )
        image_axes(ax, theme)
        ax.set_title(title, fontsize=8)
    centre = 0.5 * (shape[0] - 1)
    axes[1].plot(
        centre, centre, "+", color=theme.ink, markersize=9, markeredgewidth=1.2,
    )
    axes[1].annotate(
        r"$(x_{\rm ref}, y_{\rm ref})$: level $b$", xy=(centre, centre), xytext=(centre - 46, centre + 36),
        color=theme.ink, fontsize=7.5,
        arrowprops=dict(arrowstyle="-", color=theme.ink, linewidth=0.7),
    )
    direction = np.array([0.008, -0.005])
    direction = 22.0 * direction / np.linalg.norm(direction)
    axes[1].annotate(
        "", xy=(centre + direction[0], centre + direction[1]), xytext=(centre, centre),
        arrowprops=dict(arrowstyle="-|>", color=theme.ink, linewidth=1.0),
    )
    bar = fig.colorbar(handle, ax=axes, fraction=0.035, pad=0.03)
    bar.outline.set_edgecolor(theme.axis)
    bar.ax.tick_params(colors=theme.muted, labelcolor=theme.secondary)
    bar.set_label("residual background", color=theme.secondary)
    return fig


# ---------------------------------------------------------------------------
# Degeneracies
# ---------------------------------------------------------------------------


def _profile_of(model, shape, rmax):
    return radial_profile(model.render(shape), 60.0, 60.0, rmax)


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

    fig, (left, right) = plt.subplots(1, 2, figsize=(WIDTH, 2.7))
    fig.subplots_adjust(wspace=0.4)
    first, second = theme.series[0], theme.series[1]
    left.plot(r, reference_profile, color=first, label="n = 4, $R_e$ = 10 px, F = 100")
    left.plot(
        r, matched_profile, color=second,
        label=f"n = 3, $R_e$ = {re:.1f} px, F = {flux:.1f}",
    )
    left.set_yscale("log")
    left.set_xlim(0, rmax)
    left.set_xlabel("radius (pixels)")
    left.set_ylabel("mean intensity")
    left.set_title("Two profiles after the PSF", loc="left")
    left.legend(loc="upper right", handlelength=1.4)
    difference = 100.0 * (matched_profile - reference_profile) / reference_profile
    right.plot(r, difference, color=theme.ink)
    right.axhline(0.0, color=theme.axis, linewidth=0.8)
    right.set_xlim(0, rmax)
    right.set_xlabel("radius (pixels)")
    right.set_ylabel("difference (percent)")
    right.set_title("n = 3 relative to n = 4", loc="left")
    for ax in (left, right):
        plot_axes(ax)
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
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.6), sharey=False)
    fig.subplots_adjust(wspace=0.42)
    bulge_colour = theme.series[ROLE_SLOT["bulge"]]
    disk_colour = theme.series[ROLE_SLOT["disk"]]
    for ax, model, title in (
        (axes[0], first, "B/T = 0.30"),
        (axes[1], alternative, f"B/T = {bulge_fraction:.2f}"),
    ):
        parts = model.render_components(shape)
        for name, colour, label in (
            ("bulge", bulge_colour, "bulge"),
            ("disk", disk_colour, "disc"),
        ):
            r, profile = radial_profile(parts[name], 60.0, 60.0, rmax)
            ax.plot(r, profile, color=colour, label=label)
        r, profile = radial_profile(parts["bulge"] + parts["disk"], 60.0, 60.0, rmax)
        ax.plot(r, profile, color=theme.ink, linewidth=1.6, label="total")
        ax.set_yscale("log")
        ax.set_ylim(1e-3, 30)
        ax.set_xlim(0, rmax)
        ax.set_xlabel("radius (pixels)")
        ax.set_title(title, loc="left")
        plot_axes(ax)
    axes[0].set_ylabel("mean intensity")
    axes[0].legend(loc="upper right", handlelength=1.4)
    r, reference_profile = radial_profile(target, 60.0, 60.0, rmax)
    _, alternative_profile = radial_profile(alternative.render(shape), 60.0, 60.0, rmax)
    axes[2].plot(
        r, 100.0 * (alternative_profile - reference_profile) / reference_profile,
        color=theme.ink,
    )
    axes[2].axhline(0.0, color=theme.axis, linewidth=0.8)
    axes[2].set_xlim(0, rmax)
    axes[2].set_xlabel("radius (pixels)")
    axes[2].set_ylabel("difference of totals (percent)")
    axes[2].set_title("0.45 relative to 0.30", loc="left")
    plot_axes(axes[2])
    return fig


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
    "sersic_degeneracy": sersic_degeneracy,
    "bulge_disk_tradeoff": bulge_disk_tradeoff,
}


def main(argv: list[str] | None = None) -> int:
    """Write every figure, or the named ones, in both colour schemes."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("names", nargs="*", help="figures to write; all by default")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIRECTORY)
    arguments = parser.parse_args(argv)
    unknown = sorted(set(arguments.names) - set(FIGURES))
    if unknown:
        parser.error(f"unknown figures: {unknown}; choose from {sorted(FIGURES)}")
    names = arguments.names or list(FIGURES)
    for name in names:
        for theme in THEMES:
            apply_style(theme)
            path = save(FIGURES[name](theme), name, theme, arguments.output)
            print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
