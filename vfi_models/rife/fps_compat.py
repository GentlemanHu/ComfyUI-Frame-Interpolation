"""Drop-in, memory-bounded replacement for GACLove's RIFEInterpolation node.

Keeps the exact existing widget names, types, defaults and FPS time mapping so
saved ComfyUI workflows continue to work without rewiring.
This implementation uses the fork's RIFE 4.26 IFNet and compatible checkpoints.
"""

import gc
import logging
import math
import os
from pathlib import Path
import shutil
import tempfile
import threading
import urllib.request
import zipfile

import torch

from .output_buffer import allocate_output


_MODEL_LOCK = threading.RLock()
_CACHED = {"key": None, "model": None}
_MODEL_URLS = (
    "https://huggingface.co/hzwer/RIFE/resolve/main/RIFEv4.26_0921.zip",
    "https://hf-mirror.com/hzwer/RIFE/resolve/main/RIFEv4.26_0921.zip",
)


def _find_checkpoint(name):
    """Prefer the user's existing flownet.pkl, including GACLove's model file."""
    from vfi_utils import get_ckpt_container_path

    if Path(name).name != name or name != "flownet.pkl":
        raise ValueError("The legacy RIFEInterpolation model is flownet.pkl (RIFE 4.26)")

    home = Path(__file__).resolve().parents[2]
    custom_nodes = home.parent
    candidates = [
        home / "vfi_models" / "rife" / "flownet.pkl",
        Path(get_ckpt_container_path("rife")) / "flownet.pkl",
    ]
    try:
        import folder_paths
        candidates.append(Path(folder_paths.models_dir) / "rife" / name)
    except (ImportError, AttributeError):
        pass

    for sibling in custom_nodes.glob("*ComfyUI-VFI*"):
        candidates.extend((
            sibling / "rife" / "train_log" / name,
            sibling / "models" / name,
        ))
    for p in candidates:
        if p.is_file():
            return p
    return _download_checkpoint(Path(get_ckpt_container_path("rife")) / name)


def _download_checkpoint(target):
    """Download only the model member from the official RIFE 4.26 archive.

    Uses bounded-copying to disk, a same-directory temporary file and atomic
    replacement. Does not execute code or extract archive paths.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    last_error = None
    for url in _MODEL_URLS:
        archive_path = None
        temp_model = None
        try:
            logging.info("RIFEInterpolation: downloading RIFE 4.26 weights from %s", url)
            with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".zip", delete=False) as archive:
                archive_path = Path(archive.name)
                with urllib.request.urlopen(
                    urllib.request.Request(url, headers={"User-Agent": "ComfyUI-RIFE/1.0"}),
                    timeout=60,
                ) as response:
                    shutil.copyfileobj(response, archive, length=1024 * 1024)
            with zipfile.ZipFile(archive_path) as zf:
                members = [z for z in zf.infolist()
                           if Path(z.filename).name == "flownet.pkl" and not z.is_dir()]
                if len(members) != 1:
                    raise ValueError("RIFE 4.26 archive has no unique flownet.pkl model")
                if members[0].file_size > 1024**3:
                    raise ValueError("Unexpectedly large RIFE model archive entry")
                with tempfile.NamedTemporaryFile(
                    dir=target.parent, suffix=".pkl", delete=False
                ) as model_out:
                    temp_model = Path(model_out.name)
                    with zf.open(members[0]) as src:
                        shutil.copyfileobj(src, model_out, length=1024 * 1024)
            os.replace(temp_model, target)
            temp_model = None
            return target
        except Exception as exc:
            last_error = exc
            logging.warning("RIFEInterpolation: model download failed: %s", exc)
        finally:
            for item in (archive_path, temp_model):
                if item is not None:
                    try:
                        item.unlink(missing_ok=True)
                    except OSError:
                        pass
    raise RuntimeError(
        "Could not download RIFE 4.26 flownet.pkl. Place the checkpoint in "
        f"{target} or ComfyUI/models/rife/flownet.pkl. Last error: {last_error}"
    )


def _get_model(model_name, use_fp16):
    from .rife_arch import IFNet

    checkpoint = _find_checkpoint(model_name)
    key = (str(checkpoint.resolve()), checkpoint.stat().st_mtime_ns,
           bool(use_fp16))
    with _MODEL_LOCK:
        if _CACHED["key"] == key and _CACHED["model"] is not None:
            return _CACHED["model"]

        # Drop previous checkpoint/dtype variant rather than caching unbounded
        # GPU models. Cache the selected model on CPU between executions.
        _CACHED["key"] = None
        _CACHED["model"] = None
        gc.collect()
        model = IFNet(arch_ver="4.26")
        state = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if not isinstance(state, dict):
            raise RuntimeError("RIFE checkpoint is not a state-dict")
        # Official 4.26 checkpoints include training-only teacher/caltime
        # tensors absent from the inference-only network. Reject all other
        # missing/unexpected keys rather than silently using random weights.
        state = {
            name: tensor for k, tensor in state.items()
            if not (name := k.removeprefix("module.")).startswith(
                ("teacher.", "caltime.")
            )
        }
        model.load_state_dict(state, strict=True)
        model.eval().requires_grad_(False)
        if use_fp16:
            model = model.to(dtype=torch.float16)
        _CACHED.update(key=key, model=model)
        return model


class RIFEInterpolation:
    """Drop-in RIFEInterpolation node: same inputs and outputs as GACLove."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "source_fps": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 120.0,
                                    "step": 0.1, "display": "number"}),
            "target_fps": ("FLOAT", {"default": 60.0, "min": 1.0, "max": 240.0,
                                    "step": 0.1, "display": "number"}),
            "scale": ("FLOAT", {"default": 1.0, "min": 0.25, "max": 4.0,
                               "step": 0.25, "display": "number"}),
        }, "optional": {
            "model_name": (["flownet.pkl"], {"default": "flownet.pkl"}),
            "batch_size": ("INT", {"default": 8, "min": 1, "max": 32, "step": 1}),
            "use_fp16": ("BOOLEAN", {"default": True}),
        }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "interpolate"
    CATEGORY = "image/animation"
    DESCRIPTION = (
        "Compatible RIFE Frame Interpolation: legacy FPS interface with "
        "single-buffer output and automatic RAM/disk fallback."
    )

    def interpolate(self, images, source_fps, target_fps, scale,
                    model_name="flownet.pkl", batch_size=8, use_fp16=True):
        if not isinstance(images, torch.Tensor) or images.ndim != 4 or images.shape[-1] != 3:
            raise ValueError("RIFEInterpolation expects IMAGE [N,H,W,3]")
        if images.shape[0] < 1:
            raise ValueError("RIFEInterpolation requires at least one input frame")
        if not math.isfinite(source_fps) or not math.isfinite(target_fps) or source_fps <= 0 or target_fps <= 0:
            raise ValueError("FPS must be positive and finite")
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("scale must be positive and finite")
        if abs(source_fps - target_fps) < 0.01:
            return (images,)

        n, h, w, _ = images.shape
        target_count = int(n * target_fps / source_fps)
        if target_count <= 0:
            raise ValueError("Target FPS too low to produce a frame")

        # Exactly match GACLove's frame-count convention and time coordinate
        # system, including fractional ratios such as 24 -> 60 fps.
        tasks = []
        sources = []
        for out_idx in range(target_count):
            source_pos = out_idx * source_fps / target_fps
            src0 = min(int(source_pos), n - 1)
            src1 = min(src0 + 1, n - 1)
            dt = source_pos - int(source_pos) if src0 < src1 else 0.0
            if dt <= 1.0e-8:
                sources.append((out_idx, src0))
            else:
                tasks.append((out_idx, src0, src1, dt))

        # auto uses RAM for small clips, and disk-backed mapping when RAM/cgroup
        # headroom is insufficient; default is transparent to old workflows.
        output = allocate_output((target_count, h, w, 3), "auto", 2.0)
        with torch.inference_mode():
            for idx, src in sources:
                output[idx].copy_(images[src].to(device="cpu", dtype=torch.float32))

        if not tasks:
            return (output,)

        try:
            from comfy.utils import ProgressBar
            bar = ProgressBar(len(tasks))
        except (ImportError, AttributeError):
            bar = None

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        half = bool(use_fp16 and device.type == "cuda")
        dtype = torch.float16 if half else torch.float32
        chunk_size = max(1, int(batch_size))
        model = _get_model(model_name, half)

        with _MODEL_LOCK:
            try:
                model.to(device)
                pos = 0
                with torch.inference_mode():
                    while pos < len(tasks):
                        count = min(chunk_size, len(tasks) - pos)
                        current = tasks[pos:pos + count]
                        frame0 = frame1 = ts = predicted = None
                        try:
                            frame0 = torch.stack(
                                [images[i0] for _, i0, _, _ in current]
                            ).permute(0, 3, 1, 2).to(device=device, dtype=dtype)
                            frame1 = torch.stack(
                                [images[i1] for _, _, i1, _ in current]
                            ).permute(0, 3, 1, 2).to(device=device, dtype=dtype)
                            ts = torch.tensor(
                                [t for _, _, _, t in current],
                                device=device, dtype=dtype,
                            ).reshape(-1, 1, 1, 1)
                            predicted = model(
                                frame0, frame1, ts,
                                scale_list=[16 / scale, 8 / scale, 4 / scale, 2 / scale, 1 / scale],
                                training=False,
                                fastmode=True,
                                ensemble=False,
                            ).clamp_(0, 1)
                            for k, (out_idx, _, _, _) in enumerate(current):
                                output[out_idx].copy_(
                                    predicted[k].permute(1, 2, 0).to(device="cpu", dtype=torch.float32)
                                )
                            del predicted, ts, frame0, frame1
                            pos += count
                            if bar is not None:
                                bar.update(count)
                        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                            if "out of memory" not in str(exc).lower():
                                raise
                            if device.type != "cuda" or count <= 1:
                                raise RuntimeError(
                                    "RIFE GPU OOM even at batch_size=1; try lowering resolution."
                                ) from exc
                            # Discard GPU allocations from the failed batch and
                            # continue from the same task with a smaller batch.
                            predicted = ts = frame0 = frame1 = None
                            gc.collect()
                            torch.cuda.empty_cache()
                            chunk_size = max(1, count // 2)
                            logging.warning(
                                "RIFE GPU OOM; retrying batch with %s frames",
                                chunk_size,
                            )
            finally:
                # Avoid the video-interpolation model occupying VRAM during
                # subsequent H3 diffusion; retained CPU weights avoid reloads.
                model.to("cpu")
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        return (output,)
