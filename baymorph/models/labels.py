from __future__ import annotations

MODEL_PARAMETER_LABELS = {

    "single_sersic": {

        "flux": r"$F$",
        "re": r"$R_{\rm e}$",
        "n": r"$n$",
        "q": r"$q$",
        "theta": r"$\theta$",
        "x0": r"$x_0$",
        "y0": r"$y_0$",
        "sky": r"$S$",

    },

    "double_sersic": {

        "flux_bulge": r"$F_{\rm bulge}$",
        "re_bulge": r"$R_{{\rm e},b}$",
        "n_bulge": r"$n_b$",

        "flux_disk": r"$F_{\rm disk}$",
        "re_disk": r"$R_{{\rm e},d}$",
        "n_disk": r"$n_d$",

        "q": r"$q$",
        "theta": r"$\theta$",
        "x0": r"$x_0$",
        "y0": r"$y_0$",
        "sky": r"$S$",

    },

}


def parameter_names(
    model_name: str,
) -> list[str]:

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

    if model_name not in MODEL_PARAMETER_LABELS:
        raise ValueError(
            f"Unknown model: {model_name}"
        )

    return list(
        MODEL_PARAMETER_LABELS[
            model_name
        ].values()
    )