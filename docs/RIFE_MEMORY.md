# RIFE memory-safe interpolation

This fork's **RIFE VFI** node (from `vfi_models/rife/__init__.py`) now uses a
single directly-addressed FP32 output buffer rather than holding lists of
intermediate frames and copying the whole sequence with `torch.cat`.

## What changed

- Results from GPU batches are written **directly to final frame positions**.
  No intermediate per-frame CPU tensor list and no full-video final concatenation.
- Supports per-pair interpolation schedules and skip states.
- Keeps existing batch-size throughput optimization.
- `auto` output storage selects the usual fast CPU RAM tensor when there is
  sufficient memory; otherwise uses a temporary **disk-backed memory map**.
  This can reduce resident RAM pressure, at the cost of disk I/O for large clips.
- `ram` forces the original fast in-memory path; `disk` always uses mmap.
- On dtype/checkpoint/compile setting changes the old cached model is evicted.
  Identical settings still reuse the loaded model, avoiding reload latency.

## Options

- **output_storage**: `auto` (default), `ram`, or `disk`
- **reserve_ram_gb**: 2.0 by default, safety margin in auto mode.
- **RIFE_OUTPUT_DISK_DIR** environment variable: optionally point mmap files to
  a large **disk-backed** folder (e.g. `/notebooks`), not `/dev/shm` / tmpfs.

ComfyUI caches IMAGE outputs and downstream nodes may duplicate frame batches.
These changes reduce a major source of peak RAM, **but cannot guarantee that
arbitrarily long, high-resolution IMAGE sequences fit in a finite machine**.
For very long clips, stream RIFE from a video file and encode output in chunks.

The `clear_cache_after_n_frames` option calls CUDA cache cleaning, which does
**not** free live CPU frame tensors. Turning it down is not a fix for CPU OOM.

## Important: two different RIFE node packages

A workflow containing the node type **`RIFEInterpolation`** with `aux_id`
`GACLove/ComfyUI-VFI` uses a **different implementation** from this fork's
`RIFE VFI`. Installing this fork does not automatically replace that node.
To benefit from this patch, explicitly use this fork's RIFE VFI node and adjust
frame-rate/multiplier settings.

For another maintained approach, newer ComfyUI core versions also include
`FrameInterpolationModelLoader` and `FrameInterpolate`, with a preallocated
output buffer and GPU out-of-memory batch fallback. Check your installed
ComfyUI version and checkpoint format before switching.

## Testing

```bash
python -m unittest discover -s tests -p 'test_rife_memory.py' -v
```

CPU tests cover output ordering, skip lists, memory-mapped storage, automatic
fallback, batched inference orchestration and model-cache eviction. GPU
performance and peak RAM on specific machines still require on-device tests.

## Upstream sync

The fork's default branch has `.github/workflows/upstream-sync.yml` which
merges from `Fannovel16/ComfyUI-Frame-Interpolation:main` on alternating UTC
days (approximately 48h). This is a normal Git merge, never a force push.
Conflicting upstream changes will **stop the workflow** for manual resolution
rather than overwrite these fixes.
