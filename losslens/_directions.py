"""Filter-normalized random directions (Li et al., 2018). These choices are fixed on purpose."""
from __future__ import annotations

import torch


def trainable_params(model) -> list[torch.nn.Parameter]:
    """Parameters the landscape moves along. Frozen parameters stay fixed."""
    return [p for p in model.parameters() if p.requires_grad]


def _random_direction(params, generator: torch.Generator) -> list[torch.Tensor]:
    direction = []
    for p in params:
        if p.dim() <= 1:
            # Biases and BatchNorm parameters are not perturbed.
            direction.append(torch.zeros_like(p, memory_format=torch.contiguous_format))
            continue
        r = torch.randn(p.shape, generator=generator, dtype=torch.float32)
        r = r.to(device=p.device, dtype=p.dtype)
        # Rescale each filter (slice along dim 0) to the norm of the matching weight filter.
        w_norm = p.detach().flatten(1).norm(dim=1)
        r_norm = r.flatten(1).norm(dim=1)
        factor = (w_norm / (r_norm + 1e-10)).view(-1, *([1] * (p.dim() - 1)))
        direction.append(r * factor)
    return direction


def make_directions(params, seed: int) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Two directions that depend only on `seed` and the parameter shapes.

    Uses a private generator, so the global random state is untouched.
    """
    g = torch.Generator().manual_seed(seed)
    dx = _random_direction(params, g)
    dy = _random_direction(params, g)
    return dx, dy
