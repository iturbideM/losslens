"""`losslens.surface(...)` and the `Surface` result object."""
from __future__ import annotations

import datetime
import json
import time
from dataclasses import dataclass, field

import numpy as np
from tqdm.auto import tqdm

from . import _analysis, _plot
from ._data import prepare_batches
from ._directions import make_directions, trainable_params
from ._evaluate import LossEvaluator, evaluate_grid, resolve_device, working_copy


@dataclass
class Surface:
    """Loss values on a grid around a trained model.

    Z[iy, ix] is the loss at (x[ix], y[iy]); the trained model is at (0, 0).
    """
    x: np.ndarray
    y: np.ndarray
    Z: np.ndarray
    center_loss: float
    meta: dict = field(default_factory=dict)

    def __repr__(self):
        n = len(self.x)
        return (f"Surface({n}x{n} grid, scale={self.meta.get('scale')}, "
                f"center_loss={self.center_loss:.4g})")

    # ------------------------------------------------------------------ plots
    def plot(self, log=False, zmax=None, title=None):
        """2D contour plot. The red star is the trained model.

        log: show log10(loss), useful when the loss grows by orders of magnitude.
        zmax: top of the color scale (default: 95th percentile of the values).
        """
        import matplotlib.pyplot as plt

        vmin, vmax = _plot.color_range([self.Z], zmax, log)
        fig, ax = plt.subplots(figsize=(6, 5))
        filled = _plot.draw_contour(ax, self, vmin, vmax, log, title or self.meta.get("model"))
        fig.colorbar(filled, ax=ax, label=_plot.loss_label(log), format="%.3g")
        plt.close(fig)  # returned figure is displayed once by Jupyter
        return fig

    def plot_3d(self, log=False, zmax=None, title=None):
        """Interactive 3D surface (drag to rotate, scroll to zoom)."""
        import plotly.graph_objects as go

        vmin, vmax = _plot.color_range([self.Z], zmax, log)
        Z = _plot.prepare(self.Z, vmin, vmax, log)
        mid = len(self.x) // 2
        fig = go.Figure([
            go.Surface(x=self.x, y=self.y, z=Z, colorscale="Viridis", cmin=vmin, cmax=vmax,
                       colorbar=dict(title=_plot.loss_label(log))),
            go.Scatter3d(x=[0], y=[0], z=[Z[mid, mid]], mode="markers", name="trained model",
                         marker=dict(size=6, color="red", symbol="diamond")),
        ])
        fig.update_layout(
            title=title or self.meta.get("model"), height=550, showlegend=False,
            scene=dict(xaxis_title="direction 1", yaxis_title="direction 2",
                       zaxis_title=_plot.loss_label(log), zaxis=dict(range=[vmin, vmax])),
            margin=dict(l=0, r=0, t=40, b=0))
        return fig

    def plot_slice(self, log=False, zmax=None, title=None):
        """1D cuts through the trained model along each direction."""
        import matplotlib.pyplot as plt

        vmin, vmax = _plot.color_range([self.Z], zmax, log)
        Z = _plot.prepare(self.Z, vmin, vmax, log)
        mid = len(self.x) // 2
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(self.x, Z[mid, :], label="direction 1")
        ax.plot(self.y, Z[:, mid], label="direction 2")
        ax.plot(0, Z[mid, mid], marker="*", color="red", markersize=12, linestyle="none",
                label="trained model")
        ax.set_xlabel("distance from the trained model")
        ax.set_ylabel(_plot.loss_label(log))
        ax.set_title(title or self.meta.get("model"))
        ax.legend()
        plt.close(fig)
        return fig

    # ------------------------------------------------------------------ numbers
    def summary(self, show=True) -> dict:
        """Key numbers about the landscape. Prints a table and returns them as a dict."""
        values = _analysis.summarize(self)
        if show:
            width = max(len(d) for d in _analysis.DESCRIPTIONS.values())
            print(f"{self.meta.get('model')}  (scale={self.meta['scale']}, "
                  f"{self.meta['n_samples']} examples)")
            for key, desc in _analysis.DESCRIPTIONS.items():
                print(f"  {desc:<{width}}  {_analysis.fmt(values[key])}")
        return values

    # ------------------------------------------------------------------ files
    def save(self, path):
        """Save to a .npz file; read it back with losslens.load(path)."""
        np.savez(path, x=self.x, y=self.y, Z=self.Z, center_loss=self.center_loss,
                 meta=json.dumps(self.meta))


def load(path) -> Surface:
    """Load a surface saved with surf.save(path)."""
    with np.load(path, allow_pickle=False) as f:
        return Surface(x=f["x"], y=f["y"], Z=f["Z"], center_loss=float(f["center_loss"]),
                       meta=json.loads(str(f["meta"])))


def _fmt_seconds(s: float) -> str:
    if s < 1:
        return "under a second"
    if s < 90:
        return f"{s:.0f} s"
    if s < 5400:
        return f"{s / 60:.0f} min"
    return f"{s / 3600:.1f} h"


def surface(model, data, loss_fn=None, *, forward=None, scale=1.0, resolution=25,
            n_samples=2000, seed=0, device="auto", _progress=True) -> Surface:
    """Compute the loss landscape around a trained model.

    Args:
        model: the trained ``nn.Module``. It is not modified.
        data: a DataLoader, a Dataset, or a tuple of tensors like ``(X, y)``.
        loss_fn: ``loss_fn(outputs, targets)`` returning the batch-mean loss,
            e.g. ``nn.CrossEntropyLoss()``. Each batch must be ``(inputs, targets)``.
        forward: instead of ``loss_fn``, a function ``forward(model, batch) -> loss``
            for models that don't fit the (inputs, targets) pattern.
        scale: how far from the trained model to look. The plot covers
            ``-scale..+scale`` along each direction; 1.0 means each filter is
            perturbed by up to its own size.
        resolution: grid points per axis (rounded up to an odd number so the
            trained model is on the grid). 25 means 625 loss evaluations.
        n_samples: number of examples the loss is averaged over.
        seed: chooses the random directions and the data subset.
        device: "auto" (the model's device, else GPU if available), or e.g. "cuda".
    """
    if (loss_fn is None) == (forward is None):
        raise ValueError("Pass exactly one of `loss_fn` (e.g. nn.CrossEntropyLoss()) "
                         "or `forward` (a function forward(model, batch) -> loss).")
    if not scale > 0:
        raise ValueError(f"`scale` must be positive, got {scale}.")
    resolution = int(resolution)
    if resolution < 3:
        raise ValueError(f"`resolution` must be at least 3, got {resolution}.")
    if resolution % 2 == 0:
        resolution += 1
        print(f"Using resolution={resolution} (odd, so the trained model is on the grid).")

    dev = resolve_device(model, device)
    batches, n_used, on_device, note = prepare_batches(data, n_samples, seed, dev)
    if note:
        print("Note:", note)

    with working_copy(model, dev) as work:
        params = trainable_params(work)
        if not params:
            raise ValueError("The model has no trainable parameters (all have requires_grad=False).")
        base = [p.detach().clone() for p in params]
        dx, dy = make_directions(params, seed)
        evaluate = LossEvaluator(work, batches, loss_fn, forward, dev, on_device)

        center = evaluate()  # first call also warms up the GPU
        if not np.isfinite(center):
            raise ValueError(f"The loss of the trained model itself is {center}. "
                             "Check the model, the data and the loss function.")
        t0 = time.perf_counter()
        evaluate()
        per_point = time.perf_counter() - t0

        coords = np.linspace(-scale, scale, resolution)
        n_points = resolution * resolution
        if _progress:
            print(f"Evaluating {n_points} points on {n_used} examples "
                  f"(estimated time: {_fmt_seconds(per_point * n_points)}).")
        bar = tqdm(total=n_points, desc="losslens", disable=not _progress)
        try:
            Z = evaluate_grid(evaluate, params, base, dx, dy, coords, bar)
        finally:
            bar.close()

    meta = {
        "scale": float(scale), "resolution": resolution, "n_samples": n_used, "seed": seed,
        "model": type(model).__name__, "date": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    return Surface(x=coords, y=coords.copy(), Z=Z, center_loss=float(center), meta=meta)
