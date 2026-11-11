"""RAM-aware RIFE output buffering.

Preallocate a single FP32 output tensor; automatically use a disk-backed mmap
when the expected output would exceed the available host/cgroup RAM headroom.
"""

import os
import shutil
import tempfile
from typing import Optional, Sequence, Tuple

import numpy as np
import torch

_GIB = 1024 ** 3


def output_layout(num_frames: int, multipliers: Sequence[int], skipped: Sequence[bool]):
    """Original-frame positions, (pair, timestep, output_index) tasks, total count."""
    if num_frames < 2:
        raise ValueError("RIFE needs at least two input frames")
    if len(multipliers) < num_frames - 1 or len(skipped) != num_frames - 1:
        raise ValueError("Invalid per-pair interpolation settings")
    originals, tasks = [], []
    next_index = 0
    for pair in range(num_frames - 1):
        originals.append(next_index)
        next_index += 1
        if skipped[pair]:
            continue
        multiplier = int(multipliers[pair])
        if multiplier < 1:
            raise ValueError(f"Invalid RIFE multiplier {multiplier} for pair {pair}")
        for step in range(1, multiplier):
            tasks.append((pair, step / multiplier, next_index))
            next_index += 1
    return originals, tasks, next_index + 1


def _read_int(path: str) -> Optional[int]:
    try:
        with open(path, "r", encoding="ascii") as f:
            value = f.read().strip()
        return None if value == "max" else int(value)
    except (OSError, ValueError):
        return None


def available_ram_bytes() -> Optional[int]:
    """Take the smaller of host memory availability and cgroup headroom."""
    candidates = []
    try:
        candidates.append(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (ValueError, OSError, AttributeError):
        pass
    limit = _read_int("/sys/fs/cgroup/memory.max")
    current = _read_int("/sys/fs/cgroup/memory.current")
    if limit is None:
        limit = _read_int("/sys/fs/cgroup/memory/memory.limit_in_bytes")
        current = _read_int("/sys/fs/cgroup/memory/memory.usage_in_bytes")
    if limit is not None and current is not None and 0 < limit < (1 << 60):
        candidates.append(max(0, limit - current))
    return min(candidates) if candidates else None


def _disk_directory() -> str:
    override = os.environ.get("RIFE_OUTPUT_DISK_DIR")
    if override:
        os.makedirs(override, exist_ok=True)
        return override
    try:
        import folder_paths
        return folder_paths.get_output_directory()
    except (ImportError, AttributeError):
        return tempfile.gettempdir()


def allocate_output(shape: Tuple[int, ...], storage: str = "auto", reserve_ram_gb: float = 2.0):
    """Allocate an FP32 CPU tensor; mmap for large videos without RAM duplication.

    The resulting torch tensor owns a reference to the NumPy mmap and therefore
    keeps the disk mapping alive even after the temporary file is unlinked.
    """
    if storage not in {"auto", "ram", "disk"}:
        raise ValueError(f"Invalid RIFE output storage: {storage}")
    if any(int(x) < 1 for x in shape):
        raise ValueError(f"Invalid RIFE output shape: {shape}")
    nbytes = int(np.prod(shape, dtype=np.int64)) * 4
    available = available_ram_bytes()
    reserve = max(0, int(reserve_ram_gb * _GIB))
    disk = storage == "disk" or (
        storage == "auto" and available is not None
        # Keep headroom for one additional downstream full-image copy (e.g.
        # video encoding or ComfyUI cache); prefer mmap before RAM gets tight.
        and nbytes * 2 > max(0, int(available * 0.80) - reserve)
    )
    if not disk:
        try:
            return torch.empty(shape, dtype=torch.float32, device="cpu")
        except (RuntimeError, MemoryError):
            if storage == "ram":
                raise
            disk = True
    if disk:
        directory = _disk_directory()
        free = shutil.disk_usage(directory).free
        if free < nbytes + 256 * 1024**2:
            raise MemoryError(
                f"RIFE output needs {nbytes / _GIB:.2f} GiB of disk backing; "
                f"only {free / _GIB:.2f} GiB is free in {directory}. "
                "Set RIFE_OUTPUT_DISK_DIR to a disk-backed filesystem with sufficient space."
            )
        print(f"Comfy-VFI: disk-backed output ({nbytes / _GIB:.2f} GiB) in {directory}")
        with tempfile.TemporaryFile(prefix="rife-vfi-", dir=directory) as f:
            f.truncate(nbytes)
            mapped = np.memmap(f, dtype=np.float32, mode="r+", shape=shape)
            return torch.from_numpy(mapped)
