"""Parameter names and display labels for built-in BayMorph models.

The tables list morphology parameters only. Residual-background
parameters are owned by ``fit.background`` and labelled through
:func:`latex_parameter_label`.
"""

from __future__ import annotations

MODEL_PARAMETER_LABELS = {
    "single_sersic": {
        "flux": r"$F$",
        "re": r"$R_{e}$",
        "n": r"$n$",
        "q": r"$q$",
        "theta": r"$\theta$",
        "x0": r"$x_0$",
        "y0": r"$y_0$",
    },
    "double_sersic": {
        "flux_bulge": r"$F_{\mathrm{b}}$",
        "re_bulge": r"$R_{e,\mathrm{b}}$",
        "n_bulge": r"$n_b$",
        "q_bulge": r"$q_b$",
        "theta_bulge": r"$\theta_b$",
        "flux_disk": r"$F_{\mathrm{d}}$",
        "re_disk": r"$R_{e,\mathrm{d}}$",
        "n_disk": r"$n_d$",
        "q_disk": r"$q_d$",
        "theta_disk": r"$\theta_d$",
        "x0": r"$x_0$",
        "y0": r"$y_0$",
    },
    "sersic_point_source": {
        "flux_sersic": r"$F_{\mathrm{Sersic}}$",
        "re": r"$R_e$",
        "n": r"$n$",
        "q": r"$q$",
        "theta": r"$\theta$",
        "flux_point_source": r"$F_{\mathrm{PS}}$",
        "x0": r"$x_0$",
        "y0": r"$y_0$",
    },
    "bulge_disk": {
        "flux_bulge": r"$F_{\mathrm{b}}$",
        "re_bulge": r"$R_{e,\mathrm{b}}$",
        "n_bulge": r"$n_{\mathrm{b}}$",
        "q_bulge": r"$q_{\mathrm{b}}$",
        "theta_bulge": r"$\theta_{\mathrm{b}}$",
        "flux_disk": r"$F_{\mathrm{d}}$",
        "re_disk": r"$R_{e,\mathrm{d}}$",
        "q_disk": r"$q_{\mathrm{d}}$",
        "theta_disk": r"$\theta_{\mathrm{d}}$",
        "x0": r"$x_0$",
        "y0": r"$y_0$",
    },
}


def parameter_names(
    model_name: str,
) -> list[str]:
    """Return the ordered parameter names for a built-in model.

    Parameters
    ----------
    model_name : str
        Name of the built-in model.

    Returns
    -------
    list of str
        Ordered parameter names.

    Raises
    ------
    ValueError
        If ``model_name`` is not a registered built-in model.
    """
    if model_name not in MODEL_PARAMETER_LABELS:
        raise ValueError(
            f"Unknown model: {model_name}"
        )

    return list(
        MODEL_PARAMETER_LABELS[
            model_name
        ].keys()
    )


def parameter_labels(
    model_name: str,
) -> list[str]:
    """Return the ordered display labels for a built-in model.

    Parameters
    ----------
    model_name : str
        Name of the built-in model.

    Returns
    -------
    list of str
        Ordered parameter labels suitable for plots and reports.

    Raises
    ------
    ValueError
        If ``model_name`` is not a registered built-in model.
    """
    if model_name not in MODEL_PARAMETER_LABELS:
        raise ValueError(
            f"Unknown model: {model_name}"
        )

    return list(
        MODEL_PARAMETER_LABELS[
            model_name
        ].values()
    )


def latex_parameter_label(
    name: str,
) -> str:
    """Return the LaTeX display label for a parameter name."""
    name = str(name)

    if name.startswith("$") and name.endswith("$"):
        return name

    background_labels = {
        "background_level": r"$B$",
        "background_gradient_x": r"$B_x$",
        "background_gradient_y": r"$B_y$",
    }

    if name in background_labels:
        return background_labels[
            name
        ]

    for labels in MODEL_PARAMETER_LABELS.values():
        if name in labels:
            return labels[name]

    safe_name = name.replace("_", r"\_")

    return rf"$\mathrm{{{safe_name}}}$"
