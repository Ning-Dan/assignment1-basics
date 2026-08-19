"""Checkpoint save / load (assignment §5.2)."""

from __future__ import annotations

import os
from typing import IO, Any, BinaryIO

import torch


def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    """Return the underlying module of a ``torch.compile``d model (its state_dict keys are
    prefixed with ``_orig_mod.`` otherwise, which would not load into an uncompiled copy)."""
    return getattr(model, "_orig_mod", model)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
    extra: dict[str, Any] | None = None,
) -> None:
    """Dump model + optimizer state and the iteration number to ``out`` (path or file-like).

    ``extra`` (optional) is stored verbatim; the training script uses it for the run
    config, elapsed wall-clock time and tokens seen so a resumed run continues its curves.
    """
    obj = {
        "model": _unwrap(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": int(iteration),
    }
    if extra is not None:
        obj["extra"] = extra
    torch.save(obj, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> int:
    """Restore model (and optimizer, if given) state from ``src``; return the saved iteration."""
    obj = torch.load(src, map_location="cpu", weights_only=False)
    _unwrap(model).load_state_dict(obj["model"])
    if optimizer is not None and obj.get("optimizer") is not None:
        optimizer.load_state_dict(obj["optimizer"])
    return int(obj["iteration"])


def peek_checkpoint(src: str | os.PathLike | BinaryIO | IO[bytes]) -> dict[str, Any]:
    """Load only the metadata (iteration + extra) without touching any module."""
    obj = torch.load(src, map_location="cpu", weights_only=False)
    return {"iteration": int(obj["iteration"]), "extra": obj.get("extra", {})}
