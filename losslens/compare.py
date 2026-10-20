"""Side-by-side comparison of several surfaces."""
from __future__ import annotations

import math

from . import _analysis, _plot


def compare(surfaces: dict, log=False, zmax=None):
    """Plot several surfaces side by side on one shared color scale, and print their numbers.

    surfaces: {"name": surface, ...}, e.g. {"no skip": surf_a, "with skip": surf_b}.
    """
    import matplotlib.pyplot as plt

    if not isinstance(surfaces, dict) or len(surfaces) < 2:
        raise ValueError('Pass a dict with at least two surfaces, e.g. {"A": surf_a, "B": surf_b}.')

    for key in ("scale", "n_samples"):
        values = {s.meta.get(key) for s in surfaces.values()}
        if len(values) > 1:
            print(f"Warning: the surfaces use different `{key}` values ({sorted(values)}), "
                  "so the plots are not directly comparable.")

    names, surfs = list(surfaces), list(surfaces.values())
    vmin, vmax = _plot.color_range([s.Z for s in surfs], zmax, log)
    ncols = min(3, len(surfs))
    nrows = math.ceil(len(surfs) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.4 * nrows), squeeze=False)
    for ax, name, surf in zip(axes.flat, names, surfs):
        filled = _plot.draw_contour(ax, surf, vmin, vmax, log, title=name)
    for ax in list(axes.flat)[len(surfs):]:
        ax.axis("off")
    fig.colorbar(filled, ax=axes, label=_plot.loss_label(log), shrink=0.9, format="%.3g")

    rows = {name: _analysis.summarize(s) for name, s in surfaces.items()}
    keys = list(_analysis.DESCRIPTIONS)
    col_w = max(12, *(len(n) for n in names))
    desc_w = max(len(d) for d in _analysis.DESCRIPTIONS.values())
    print(" " * desc_w + "  " + "  ".join(f"{n:>{col_w}}" for n in names))
    for k in keys:
        cells = "  ".join(f"{_analysis.fmt(rows[n][k]):>{col_w}}" for n in names)
        print(f"{_analysis.DESCRIPTIONS[k]:<{desc_w}}  {cells}")

    plt.close(fig)
    return fig
