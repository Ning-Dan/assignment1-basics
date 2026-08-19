"""Data loading for language-model training (assignment §5.1).

The tokenized corpus is one long 1-D integer array (documents concatenated with
``<|endoftext|>`` between them).  A batch is ``batch_size`` random windows of
length ``context_length`` plus their one-step-shifted targets.  Works on plain
ndarrays and on ``np.memmap`` / ``np.load(mmap_mode="r")`` arrays, so the full
dataset never has to be materialised in RAM.
"""

from __future__ import annotations

import os

import numpy as np
import numpy.typing as npt
import torch


def load_token_file(path: str | os.PathLike, dtype: npt.DTypeLike = np.uint16) -> np.ndarray:
    """Open a token file memory-mapped (read-only).

    ``.npy`` files (written with ``np.save``) are opened with ``np.load(mmap_mode="r")``,
    which recovers the dtype from the header; any other file is treated as a raw
    ``dtype`` array and opened with ``np.memmap``.
    """
    path = os.fspath(path)
    if path.endswith(".npy"):
        arr = np.load(path, mmap_mode="r")
    else:
        arr = np.memmap(path, dtype=dtype, mode="r")
    if arr.ndim != 1:
        raise ValueError(f"expected a 1-D token array in {path}, got shape {arr.shape}")
    return arr


def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str | torch.device,
    generator: np.random.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample ``batch_size`` (input, target) windows of length ``context_length``.

    Start positions ``i`` are drawn uniformly from ``[0, len(dataset) - context_length)``
    so that ``dataset[i + context_length]`` (the last target) always exists.

    Returns two ``torch.int64`` tensors of shape ``(batch_size, context_length)`` on
    ``device``: inputs ``x = dataset[i : i + m]`` and targets ``y = dataset[i + 1 : i + m + 1]``.
    """
    n = len(dataset)
    if n <= context_length:
        raise ValueError(f"dataset has {n} tokens, need more than context_length={context_length}")
    if generator is None:
        starts = np.random.randint(0, n - context_length, size=batch_size)
    else:
        starts = generator.integers(0, n - context_length, size=batch_size)
    # One gather for x and y together: (batch_size, context_length + 1).
    idx = starts[:, None] + np.arange(context_length + 1)[None, :]
    window = np.asarray(dataset[idx], dtype=np.int64)  # copies out of the memmap
    x = torch.from_numpy(np.ascontiguousarray(window[:, :-1]))
    y = torch.from_numpy(np.ascontiguousarray(window[:, 1:]))
    return x.to(device), y.to(device)
