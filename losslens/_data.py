"""Turn whatever data the student passes into a fixed, cached list of batches."""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset, Subset, TensorDataset

DEFAULT_BATCH_SIZE = 256


def to_device(obj, device):
    """Move tensors inside (nested) tuples, lists and dicts to `device`."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        moved = [to_device(v, device) for v in obj]
        return type(obj)(moved) if not hasattr(obj, "_fields") else type(obj)(*moved)
    return obj


def batch_size_of(batch) -> int:
    """Size of the first dimension of the first tensor found in the batch."""
    if isinstance(batch, torch.Tensor):
        return batch.shape[0] if batch.dim() > 0 else 1
    if isinstance(batch, dict):
        items = batch.values()
    elif isinstance(batch, (list, tuple)):
        items = batch
    else:
        items = []
    for item in items:
        n = batch_size_of(item)
        if n:
            return n
    return 0


def _as_dataset(data):
    """Return (dataset, batch_size, collate_fn) or None if `data` must be iterated."""
    if isinstance(data, DataLoader):
        if isinstance(data.dataset, IterableDataset) or data.batch_size is None:
            return None
        return data.dataset, data.batch_size, data.collate_fn
    if isinstance(data, (tuple, list)) and data and all(isinstance(t, torch.Tensor) for t in data):
        n = len(data[0])
        if any(len(t) != n for t in data):
            raise ValueError("All tensors in `data` must have the same length "
                             f"(got {[len(t) for t in data]}).")
        return TensorDataset(*data), DEFAULT_BATCH_SIZE, None
    if isinstance(data, IterableDataset):
        return None
    if isinstance(data, Dataset) or (hasattr(data, "__getitem__") and hasattr(data, "__len__")):
        return data, DEFAULT_BATCH_SIZE, None
    raise TypeError("`data` must be a DataLoader, a Dataset, or a tuple of tensors like (X, y); "
                    f"got {type(data).__name__}.")


def prepare_batches(data, n_samples: int | None, seed: int, device):
    """Draw a fixed subset of `n_samples` examples and collate it into batches.

    Returns (batches, n_used, on_device, note). `note` is a message for the user or None.
    """
    note = None
    spec = _as_dataset(data)

    if spec is not None:
        dataset, batch_size, collate_fn = spec
        total = len(dataset)
        if total == 0:
            raise ValueError("`data` is empty.")
        n = total if n_samples is None else min(int(n_samples), total)
        g = torch.Generator().manual_seed(seed)
        indices = torch.randperm(total, generator=g)[:n].sort().values.tolist()
        loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False,
                            num_workers=0, **({"collate_fn": collate_fn} if collate_fn else {}))
        batches = list(loader)
    else:
        # Iterable data: we cannot sample, so take the first n_samples examples.
        batches, n = [], 0
        for batch in data:
            batches.append(batch)
            n += batch_size_of(batch)
            if n_samples is not None and n >= n_samples:
                break
        if not batches:
            raise ValueError("`data` produced no batches.")
        note = (f"`data` can't be sampled randomly, so the first {n} examples it produced are "
                "used. Make sure they are representative (e.g. not sorted by class).")

    try:
        batches = [to_device(b, device) for b in batches]
        on_device = True
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        on_device = False
        note = ((note + " ") if note else "") + (
            "The evaluation data doesn't fit in GPU memory, so it is moved batch by batch "
            "(slower). Lower `n_samples` to speed this up.")
    return batches, n, on_device, note
