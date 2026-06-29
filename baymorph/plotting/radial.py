import matplotlib.pyplot as plt

from .style import style_axes


def plot_radial_profile_diagnostics(
    radius,
    profile_data,
    profile_model,
    profile_noise=None,
    ax=None,
):
    """
    Plot radial profile and residuals.

    Parameters
    ----------
    radius : array
    profile_data : array
    profile_model : array
    profile_noise : array, optional
    ax : matplotlib.axes.Axes, optional
    """

    if ax is None:
        fig, (ax1, ax2) = plt.subplots(
            2,
            1,
            figsize=(15, 15),
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1]},
        )
    else:
        fig = ax.figure
        gs = ax.get_subplotspec().subgridspec(
            2,
            1,
            height_ratios=[3, 1],
            hspace=0.05,
        )
        ax.remove()
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1], sharex=ax1)

    ax1.plot(radius, profile_data, "ko", ms=3, label="Data")
    ax1.plot(radius, profile_model, "r-", lw=2, label="Model")

    if profile_noise is not None:
        ax1.fill_between(
            radius,
            profile_model - profile_noise,
            profile_model + profile_noise,
            color="0.8",
            alpha=0.5,
        )

    residual = profile_data - profile_model

    ax2.axhline(
        0,
        color="k",
        lw=1,
    )

    ax2.plot(
        radius,
        residual,
        "ko",
        ms=3,
    )

    ax1.set_ylabel("Intensity")
    ax2.set_ylabel("Residual")
    ax2.set_xlabel("Radius [pix]")

    # Hide x tick labels on the upper panel
    ax1.tick_params(
        axis="x",
        which="both",
        labelbottom=False,
    )

    ax1.legend(
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
        frameon=False,
    )

    style_axes(ax1)
    style_axes(ax2)

    ax1.tick_params(
        labelbottom=False,
    )

    ax1.tick_params(
        top=False,
    )

    ax2.tick_params(
        top=False,
    )

    return fig, (ax1, ax2)