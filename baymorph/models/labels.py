"""Parameter names, display labels and kinds for built-in BayMorph models.

The tables list morphology parameters only. Residual-background
parameters are owned by ``fit.background`` and labelled through
:func:`latex_parameter_label`.
"""

from __future__ import annotations

PARAMETER_KIND_NAMES = (
    "flux",
    "length",
    "position",
    "angle",
    "dimensionless",
)
"""Kinds of model parameters: what a parameter measures, which decides how a
prior in physical units maps to it (AD-049)."""

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
    "boxy_sersic": {
        "flux": r"$F$",
        "re": r"$R_{e}$",
        "n": r"$n$",
        "q": r"$q$",
        "c0": r"$c_0$",
        "theta": r"$\theta$",
        "x0": r"$x_0$",
        "y0": r"$y_0$",
    },
    "bulge_broken_disk": {
        "flux_bulge": r"$F_{\mathrm{b}}$",
        "re_bulge": r"$R_{e,\mathrm{b}}$",
        "n_bulge": r"$n_{\mathrm{b}}$",
        "q_bulge": r"$q_{\mathrm{b}}$",
        "theta_bulge": r"$\theta_{\mathrm{b}}$",
        "flux_disk": r"$F_{\mathrm{d}}$",
        "h_in_disk": r"$h_{\mathrm{in}}$",
        "h_out_disk": r"$h_{\mathrm{out}}$",
        "rb_disk": r"$R_{\mathrm{br}}$",
        "wb_disk": r"$w_{\mathrm{br}}$",
        "q_disk": r"$q_{\mathrm{d}}$",
        "theta_disk": r"$\theta_{\mathrm{d}}$",
        "x0": r"$x_0$",
        "y0": r"$y_0$",
    },
    "bulge_disk_bar": {
        "flux_bulge": r"$F_{\mathrm{b}}$",
        "re_bulge": r"$R_{e,\mathrm{b}}$",
        "n_bulge": r"$n_{\mathrm{b}}$",
        "q_bulge": r"$q_{\mathrm{b}}$",
        "theta_bulge": r"$\theta_{\mathrm{b}}$",
        "flux_disk": r"$F_{\mathrm{d}}$",
        "re_disk": r"$R_{e,\mathrm{d}}$",
        "q_disk": r"$q_{\mathrm{d}}$",
        "theta_disk": r"$\theta_{\mathrm{d}}$",
        "flux_bar": r"$F_{\mathrm{bar}}$",
        "a_bar": r"$a_{\mathrm{bar}}$",
        "alpha_bar": r"$\alpha_{\mathrm{bar}}$",
        "c0_bar": r"$c_{0,\mathrm{bar}}$",
        "q_bar": r"$q_{\mathrm{bar}}$",
        "theta_bar": r"$\theta_{\mathrm{bar}}$",
        "x0": r"$x_0$",
        "y0": r"$y_0$",
    },
    "bulge_disk_ring": {
        "flux_bulge": r"$F_{\mathrm{b}}$",
        "re_bulge": r"$R_{e,\mathrm{b}}$",
        "n_bulge": r"$n_{\mathrm{b}}$",
        "q_bulge": r"$q_{\mathrm{b}}$",
        "theta_bulge": r"$\theta_{\mathrm{b}}$",
        "flux_disk": r"$F_{\mathrm{d}}$",
        "re_disk": r"$R_{e,\mathrm{d}}$",
        "q_disk": r"$q_{\mathrm{d}}$",
        "theta_disk": r"$\theta_{\mathrm{d}}$",
        "flux_ring": r"$F_{\mathrm{ring}}$",
        "r_ring": r"$R_{\mathrm{ring}}$",
        "sigma_ring": r"$\sigma_{\mathrm{ring}}$",
        "q_ring": r"$q_{\mathrm{ring}}$",
        "theta_ring": r"$\theta_{\mathrm{ring}}$",
        "x0": r"$x_0$",
        "y0": r"$y_0$",
    },
    "bulge_disk_spiral": {
        "flux_bulge": r"$F_{\mathrm{b}}$",
        "re_bulge": r"$R_{e,\mathrm{b}}$",
        "n_bulge": r"$n_{\mathrm{b}}$",
        "q_bulge": r"$q_{\mathrm{b}}$",
        "theta_bulge": r"$\theta_{\mathrm{b}}$",
        "flux_disk": r"$F_{\mathrm{d}}$",
        "re_disk": r"$R_{e,\mathrm{d}}$",
        "q_disk": r"$q_{\mathrm{d}}$",
        "theta_disk": r"$\theta_{\mathrm{d}}$",
        "amplitude_spiral": r"$A_{0}$",
        "winding_spiral": r"$w$",
        "phase_spiral": r"$\varphi_{0}$",
        "r_in_spiral": r"$r_{\mathrm{in}}$",
        "x0": r"$x_0$",
        "y0": r"$y_0$",
    },
    "core_sersic": {
        "flux": r"$F$",
        "re": r"$R_{e}$",
        "n": r"$n$",
        "rb": r"$R_{b}$",
        "gamma": r"$\gamma$",
        "alpha": r"$\alpha$",
        "q": r"$q$",
        "theta": r"$\theta$",
        "x0": r"$x_0$",
        "y0": r"$y_0$",
    },
}


_SHARED_CENTRE_KINDS = {
    "x0": "position",
    "y0": "position",
}

PARAMETER_KINDS = {
    "single_sersic": {
        "flux": "flux",
        "re": "length",
        "n": "dimensionless",
        "q": "dimensionless",
        "theta": "angle",
        **_SHARED_CENTRE_KINDS,
    },
    "double_sersic": {
        "flux_bulge": "flux",
        "re_bulge": "length",
        "n_bulge": "dimensionless",
        "q_bulge": "dimensionless",
        "theta_bulge": "angle",
        "flux_disk": "flux",
        "re_disk": "length",
        "n_disk": "dimensionless",
        "q_disk": "dimensionless",
        "theta_disk": "angle",
        **_SHARED_CENTRE_KINDS,
    },
    "sersic_point_source": {
        "flux_sersic": "flux",
        "re": "length",
        "n": "dimensionless",
        "q": "dimensionless",
        "theta": "angle",
        "flux_point_source": "flux",
        **_SHARED_CENTRE_KINDS,
    },
    "bulge_disk": {
        "flux_bulge": "flux",
        "re_bulge": "length",
        "n_bulge": "dimensionless",
        "q_bulge": "dimensionless",
        "theta_bulge": "angle",
        "flux_disk": "flux",
        "re_disk": "length",
        "q_disk": "dimensionless",
        "theta_disk": "angle",
        **_SHARED_CENTRE_KINDS,
    },
    "boxy_sersic": {
        "flux": "flux",
        "re": "length",
        "n": "dimensionless",
        "q": "dimensionless",
        "c0": "dimensionless",
        "theta": "angle",
        **_SHARED_CENTRE_KINDS,
    },
    "bulge_broken_disk": {
        "flux_bulge": "flux",
        "re_bulge": "length",
        "n_bulge": "dimensionless",
        "q_bulge": "dimensionless",
        "theta_bulge": "angle",
        "flux_disk": "flux",
        "h_in_disk": "length",
        "h_out_disk": "length",
        "rb_disk": "length",
        "wb_disk": "length",
        "q_disk": "dimensionless",
        "theta_disk": "angle",
        **_SHARED_CENTRE_KINDS,
    },
    "bulge_disk_bar": {
        "flux_bulge": "flux",
        "re_bulge": "length",
        "n_bulge": "dimensionless",
        "q_bulge": "dimensionless",
        "theta_bulge": "angle",
        "flux_disk": "flux",
        "re_disk": "length",
        "q_disk": "dimensionless",
        "theta_disk": "angle",
        "flux_bar": "flux",
        "a_bar": "length",
        "alpha_bar": "dimensionless",
        "c0_bar": "dimensionless",
        "q_bar": "dimensionless",
        "theta_bar": "angle",
        **_SHARED_CENTRE_KINDS,
    },
    "bulge_disk_ring": {
        "flux_bulge": "flux",
        "re_bulge": "length",
        "n_bulge": "dimensionless",
        "q_bulge": "dimensionless",
        "theta_bulge": "angle",
        "flux_disk": "flux",
        "re_disk": "length",
        "q_disk": "dimensionless",
        "theta_disk": "angle",
        "flux_ring": "flux",
        "r_ring": "length",
        "sigma_ring": "length",
        "q_ring": "dimensionless",
        "theta_ring": "angle",
        **_SHARED_CENTRE_KINDS,
    },
    "bulge_disk_spiral": {
        "flux_bulge": "flux",
        "re_bulge": "length",
        "n_bulge": "dimensionless",
        "q_bulge": "dimensionless",
        "theta_bulge": "angle",
        "flux_disk": "flux",
        "re_disk": "length",
        "q_disk": "dimensionless",
        "theta_disk": "angle",
        "amplitude_spiral": "dimensionless",
        "winding_spiral": "dimensionless",
        "phase_spiral": "angle",
        "r_in_spiral": "length",
        **_SHARED_CENTRE_KINDS,
    },
    "core_sersic": {
        "flux": "flux",
        "re": "length",
        "n": "dimensionless",
        "rb": "length",
        "gamma": "dimensionless",
        "alpha": "dimensionless",
        "q": "dimensionless",
        "theta": "angle",
        **_SHARED_CENTRE_KINDS,
    },
}
"""Kind of every parameter of the built-in models: a flux in image units, a
length in pixels, a position in pixels, an angle in radians, or a
dimensionless number. The advisor of AD-049 maps priors in physical units to
parameters through their kind."""


def parameter_kind(
    model_name: str,
    parameter: str,
) -> str:
    """Return the kind of one parameter of a built-in model.

    Parameters
    ----------
    model_name : str
        Name of the built-in model.
    parameter : str
        Name of one of its parameters.

    Returns
    -------
    str
        One of ``PARAMETER_KIND_NAMES``.

    Raises
    ------
    ValueError
        If the model or the parameter is unknown.
    """
    kinds = PARAMETER_KINDS.get(model_name)

    if kinds is None:
        raise ValueError(f"Unknown model: {model_name}")

    if parameter not in kinds:
        raise ValueError(
            f"Model {model_name!r} has no parameter {parameter!r}."
        )

    return kinds[parameter]


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
