"""Tests for canonical Computer Modern plotting typography."""

from __future__ import annotations

from pathlib import Path
import shutil
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties

from baymorph.plotting import corner_plot
from baymorph.plotting.images import plot_log_image
from baymorph.plotting.style import (
    computer_modern_text,
    make_plot,
    set_plot_style,
)


DISPLAY_TITLE = "Simulated single-Sérsic observation"
RENDERED_TITLE = (
    r"Simulated single-S$\acute{\mathrm{e}}$rsic observation"
)


def _latex_on_path(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "baymorph.plotting.style.shutil.which",
        lambda name: f"/usr/bin/{name}",
    )


def test_latex_is_the_default_typesetting(
    monkeypatch,
) -> None:
    """Without an explicit choice, figures are typeset with LaTeX."""
    monkeypatch.delenv(
        "BAYMORPH_USETEX",
        raising=False,
    )
    _latex_on_path(
        monkeypatch
    )

    with plt.rc_context():
        set_plot_style()

        assert plt.rcParams["text.usetex"] is True
        assert plt.rcParams["font.serif"][0] == "Computer Modern Roman"


SPECIAL_TEXT = r"68% & #1 | r < 2 > 1, $50\%$ \& done"


@pytest.mark.parametrize(
    ("usetex", "expected"),
    [
        (
            True,
            r"68\% \& \#1 $|$ r $<$ 2 $>$ 1, $50\%$ \& done",
        ),
        (
            False,
            SPECIAL_TEXT,
        ),
    ],
)
def test_latex_special_characters_are_adapted_only_with_latex(
    usetex: bool,
    expected: str,
) -> None:
    """With LaTeX, %, &, #, |, < and > outside math print as written."""
    with plt.rc_context(
        {
            "text.usetex": usetex,
        }
    ):
        assert (
            computer_modern_text(
                SPECIAL_TEXT
            )
            == expected
        )


@pytest.mark.skipif(
    shutil.which("latex") is None or shutil.which("dvipng") is None,
    reason="LaTeX is not installed.",
)
def test_figure_text_renders_with_latex(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """A figure with special characters must typeset with real LaTeX."""
    monkeypatch.setenv(
        "BAYMORPH_USETEX",
        "1",
    )

    with plt.rc_context():
        fig, ax = make_plot()
        ax.plot(
            [0, 1],
            [0, 1],
            label=computer_modern_text(
                "68% posterior interval"
            ),
        )
        ax.legend()
        ax.set_title(
            computer_modern_text(
                "single_sersic | observation: r < 25 & #1, "
                + DISPLAY_TITLE
            )
        )
        fig.savefig(
            tmp_path / "latex.png",
            dpi=10,
        )
        plt.close(fig)

    assert (tmp_path / "latex.png").stat().st_size > 0


def test_missing_latex_fails_with_a_clear_error(
    monkeypatch,
) -> None:
    """LaTeX without its executables must fail, naming the way out."""
    monkeypatch.delenv(
        "BAYMORPH_USETEX",
        raising=False,
    )
    monkeypatch.setattr(
        "baymorph.plotting.style.shutil.which",
        lambda name: None,
    )

    with plt.rc_context():
        with pytest.raises(
            RuntimeError,
            match="latex and dvipng.*BAYMORPH_USETEX=0",
        ):
            make_plot()


@pytest.mark.parametrize(
    ("environment", "usetex"),
    [
        ("0", None),
        ("1", False),
    ],
)
def test_latex_is_turned_off_only_explicitly(
    monkeypatch,
    environment: str,
    usetex: bool | None,
) -> None:
    """BAYMORPH_USETEX=0 or usetex=False select the bundled mathtext."""
    monkeypatch.setenv(
        "BAYMORPH_USETEX",
        environment,
    )
    monkeypatch.setattr(
        "baymorph.plotting.style.shutil.which",
        lambda name: None,
    )

    with plt.rc_context():
        set_plot_style(
            usetex=usetex,
        )

        assert plt.rcParams["text.usetex"] is False
        assert plt.rcParams["font.serif"][0] == "cmr10"


def test_plot_style_uses_bundled_computer_modern(
    monkeypatch,
) -> None:
    """Without LaTeX, all text and mathematical symbols must use CM fonts."""
    monkeypatch.setenv(
        "BAYMORPH_USETEX",
        "0",
    )
    set_plot_style()

    assert plt.rcParams["font.family"] == [
        "serif",
    ]
    assert plt.rcParams["font.serif"][0] == "cmr10"
    assert plt.rcParams["mathtext.fontset"] == "cm"
    assert plt.rcParams["axes.formatter.use_mathtext"]

    resolved_font = Path(
        font_manager.findfont(
            FontProperties(
                family="serif",
            )
        )
    ).name

    assert resolved_font == "cmr10.ttf"


def test_computer_modern_text_preserves_accent_visually() -> None:
    """Unsupported Unicode accents must become roman CM MathText."""
    assert (
        computer_modern_text(
            DISPLAY_TITLE
        )
        == RENDERED_TITLE
    )


def test_log_image_renders_accent_without_font_warning() -> None:
    """Natural Unicode titles must render without missing CM glyphs."""
    with warnings.catch_warnings():
        warnings.simplefilter(
            "error",
            UserWarning,
        )

        fig, ax = plot_log_image(
            np.ones(
                (
                    5,
                    5,
                )
            ),
            title=DISPLAY_TITLE,
        )

        fig.canvas.draw()

    assert ax.get_title() == RENDERED_TITLE

    plt.close(
        fig
    )


def test_corner_renders_accent_without_font_warning() -> None:
    """Corner titles must use the same Computer Modern conversion."""
    rng = np.random.default_rng(
        42
    )

    with warnings.catch_warnings():
        warnings.simplefilter(
            "error",
            UserWarning,
        )

        fig, _, _ = corner_plot(
            rng.normal(
                size=(
                    3,
                    200,
                )
            ),
            parameter_labels=[
                r"$F$",
                r"$R_e$",
                r"$n$",
            ],
            title=DISPLAY_TITLE,
        )

        fig.canvas.draw()

    rendered_titles = [
        text.get_text()
        for text in fig.texts
    ]

    assert RENDERED_TITLE in rendered_titles

    plt.close(
        fig
    )
