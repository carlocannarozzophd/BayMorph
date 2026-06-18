from __future__ import annotations

import numpy as np

from baymorph import SersicModel


def render_components_from_values(
    model,
    values,
    shape,
):
    values = np.asarray(values, dtype=float)

    if values.size != 10:
        raise ValueError("Expected 10 parameters for double-Sersic components.")

    x0 = model.bulge.x0
    y0 = model.bulge.y0

    inner = SersicModel(
        flux=values[0],
        re=values[1],
        n=values[2],
        q=values[3],
        theta=values[4],
        x0=x0,
        y0=y0,
    ).render(shape=shape)

    outer = SersicModel(
        flux=values[5],
        re=values[6],
        n=values[7],
        q=values[8],
        theta=values[9],
        x0=x0,
        y0=y0,
    ).render(shape=shape)

    total = inner + outer

    return inner, outer, total