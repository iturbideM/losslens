"""Numbers for summary() and compare()."""
from __future__ import annotations

import numpy as np

RADII = (0.1, 0.25, 0.5)  # as fractions of `scale`


def _bilinear(x, Z, px, py):
    """Interpolate Z (indexed [iy, ix] on the square grid x) at points (px, py)."""
    step = x[1] - x[0]
    fx = (np.asarray(px) - x[0]) / step
    fy = (np.asarray(py) - x[0]) / step
    i0 = np.clip(np.floor(fx).astype(int), 0, len(x) - 2)
    j0 = np.clip(np.floor(fy).astype(int), 0, len(x) - 2)
    tx, ty = fx - i0, fy - j0
    return ((1 - tx) * (1 - ty) * Z[j0, i0] + tx * (1 - ty) * Z[j0, i0 + 1]
            + (1 - tx) * ty * Z[j0 + 1, i0] + tx * ty * Z[j0 + 1, i0 + 1])


def summarize(surf) -> dict:
    x, Z, scale = surf.x, surf.Z, surf.meta["scale"]
    center = surf.center_loss
    out = {"center_loss": center}

    iy, ix = np.unravel_index(np.argmin(Z), Z.shape)
    out["grid_min_loss"] = float(Z[iy, ix])
    out["grid_min_at"] = (float(x[ix]), float(surf.y[iy]))
    out["min_is_center"] = bool(ix == len(x) // 2 and iy == len(x) // 2)

    angles = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    for frac in RADII:
        r = frac * scale
        ring = _bilinear(x, Z, r * np.cos(angles), r * np.sin(angles))
        out[f"rise_at_{int(frac * 100)}%"] = float(np.mean(ring) - center)

    # How differently the loss rises along the two directions, at half the scale.
    r = 0.5 * scale
    rise_1 = np.mean(_bilinear(x, Z, [-r, r], [0, 0])) - center
    rise_2 = np.mean(_bilinear(x, Z, [0, 0], [-r, r])) - center
    lo, hi = sorted([rise_1, rise_2])
    out["asymmetry"] = float(hi / lo) if lo > 0 else float("inf")
    return out


DESCRIPTIONS = {
    "center_loss": "loss of the trained model",
    "grid_min_loss": "lowest loss on the grid",
    "grid_min_at": "where it is (direction 1, direction 2)",
    "min_is_center": "is the trained model the lowest point?",
    "rise_at_10%": "mean loss increase at 10% of scale",
    "rise_at_25%": "mean loss increase at 25% of scale",
    "rise_at_50%": "mean loss increase at 50% of scale",
    "asymmetry": "steeper / flatter direction, at 50% of scale",
}


def fmt(value) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, tuple):
        return "(" + ", ".join(f"{v:+.2f}" for v in value) + ")"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)
