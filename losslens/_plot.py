"""Plotting helpers shared by Surface and compare()."""
from __future__ import annotations

import numpy as np

CLIP_PERCENTILE = 95
LEVELS = 20


def color_range(Zs, zmax=None, log=False):
    """(vmin, vmax) shared by a list of grids, after the optional log transform.

    By default the top of the scale is the 95th percentile of the finite values, so a few
    exploding corners don't wash out the rest of the plot.
    """
    finite = np.concatenate([Z[np.isfinite(Z)].ravel() for Z in Zs])
    vmin = float(finite.min())
    vmax = float(zmax) if zmax is not None else float(np.percentile(finite, CLIP_PERCENTILE))
    if vmax <= vmin:
        vmax = float(finite.max()) if finite.max() > vmin else vmin + 1e-6
    if log:
        floor = _log_floor(finite)
        vmin, vmax = np.log10(max(vmin, floor)), np.log10(max(vmax, floor))
    return vmin, vmax


def _log_floor(values):
    positive = values[values > 0]
    return float(positive.min()) if positive.size else 1e-12


def prepare(Z, vmin, vmax, log=False):
    """Transform and clip Z for display. inf (diverged points) shows as the top of the scale."""
    Z = np.where(np.isfinite(Z), Z, np.inf)
    if log:
        finite = Z[np.isfinite(Z)]
        Z = np.log10(np.maximum(Z, _log_floor(finite)))
    return np.clip(Z, vmin, vmax)


def loss_label(log):
    return "log10(loss)" if log else "loss"


def draw_contour(ax, surf, vmin, vmax, log=False, title=None):
    Z = prepare(surf.Z, vmin, vmax, log)
    levels = np.linspace(vmin, vmax, LEVELS + 1)
    filled = ax.contourf(surf.x, surf.y, Z, levels=levels, cmap="viridis", extend="neither")
    lines = ax.contour(surf.x, surf.y, Z, levels=levels[::4], colors="white",
                       linewidths=0.6, alpha=0.7)
    ax.clabel(lines, inline=True, fontsize=7, fmt="%.3g")
    ax.plot(0, 0, marker="*", color="red", markersize=12, markeredgecolor="white",
            label="trained model")
    ax.set_xlabel("direction 1")
    ax.set_ylabel("direction 2")
    ax.set_aspect("equal")
    if title:
        ax.set_title(title)
    return filled
