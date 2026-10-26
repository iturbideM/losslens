"""Loss at one point in weight space, and the grid loop."""
from __future__ import annotations

import contextlib
import copy
import math

import numpy as np
import torch

from ._data import batch_size_of, to_device


def resolve_device(model, device):
    if device != "auto":
        return torch.device(device)
    for p in model.parameters():
        if p.device.type != "cpu":
            return p.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@contextlib.contextmanager
def working_copy(model, device):
    """Yield a model we may freely modify; the caller's model is left exactly as it was."""
    try:
        work = copy.deepcopy(model)
        restore = None
    except Exception:
        # Some models can't be deep-copied: modify the original and put everything back after.
        work = model
        first = next(model.parameters(), None)
        restore = (
            {k: v.detach().clone() for k, v in model.state_dict().items()},
            model.training,
            first.device if first is not None else None,
        )
    try:
        work.to(device)
        work.eval()
        yield work
    finally:
        if restore is not None:
            state, training, orig_device = restore
            with torch.no_grad():
                model.load_state_dict(state)
            if orig_device is not None:
                model.to(orig_device)
            model.train(training)


class LossEvaluator:
    """Average loss of `model` over a fixed list of batches."""

    def __init__(self, model, batches, loss_fn, forward, device, on_device):
        self.model, self.batches = model, batches
        self.loss_fn, self.forward = loss_fn, forward
        self.device, self.on_device = device, on_device

    def _batch_loss(self, batch):
        if self.forward is not None:
            return self.forward(self.model, batch)
        if not isinstance(batch, (list, tuple)) or len(batch) < 2:
            raise TypeError(
                "With `loss_fn`, each batch must be (inputs, targets). Your batches look different "
                f"({type(batch).__name__}); pass `forward=lambda model, batch: <loss>` instead.")
        return self.loss_fn(self.model(batch[0]), batch[1])

    @torch.inference_mode()
    def __call__(self) -> float:
        total, count = 0.0, 0
        for batch in self.batches:
            if not self.on_device:
                batch = to_device(batch, self.device)
            loss = self._batch_loss(batch)
            if not isinstance(loss, torch.Tensor):
                loss = torch.as_tensor(loss)
            if loss.numel() != 1:
                raise ValueError(
                    f"The loss must be a single number, got shape {tuple(loss.shape)}. "
                    "Use a loss that averages over the batch (e.g. reduction='mean').")
            n = batch_size_of(batch) or 1
            total += loss.item() * n
            count += n
        return total / count


def set_point(params, base, dx, dy, a: float, b: float):
    """Move the model to base + a*dx + b*dy (in place)."""
    with torch.no_grad():
        for p, w, u, v in zip(params, base, dx, dy):
            p.copy_(w + a * u + b * v)


def evaluate_grid(evaluate, params, base, dx, dy, coords, progress=None) -> np.ndarray:
    """Z[iy, ix] = loss at base + coords[ix]*dx + coords[iy]*dy. Non-finite values become inf."""
    Z = np.empty((len(coords), len(coords)))
    for iy, b in enumerate(coords):
        for ix, a in enumerate(coords):
            set_point(params, base, dx, dy, float(a), float(b))
            value = evaluate()
            Z[iy, ix] = value if math.isfinite(value) else np.inf
            if progress is not None:
                progress.update(1)
    set_point(params, base, dx, dy, 0.0, 0.0)
    return Z
