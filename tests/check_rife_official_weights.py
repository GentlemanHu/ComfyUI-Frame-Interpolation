"""Optional CI smoke test against the actual upstream RIFE 4.26 flownet.pkl.

This test downloads weights only in the pull-request CI job, never at import
time or in ordinary test runs.
"""
import importlib.util
import io
from pathlib import Path
import sys
import types
import urllib.request
import zipfile

import torch

ROOT = Path(__file__).resolve().parents[1]
URLS = [
    "https://huggingface.co/hzwer/RIFE/resolve/main/RIFEv4.26_0921.zip",
    "https://hf-mirror.com/hzwer/RIFE/resolve/main/RIFEv4.26_0921.zip",
]

comfy = types.ModuleType("comfy")
model_management = types.ModuleType("comfy.model_management")
model_management.get_torch_device = lambda: torch.device("cpu")
comfy.model_management = model_management
sys.modules["comfy"] = comfy
sys.modules["comfy.model_management"] = model_management

spec = importlib.util.spec_from_file_location(
    "official_rife_arch", ROOT / "vfi_models/rife/rife_arch.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

archive = None
last_error = None
for url in URLS:
    try:
        with urllib.request.urlopen(
            urllib.request.Request(url, headers={"User-Agent": "ComfyUI-RIFE/1.0"}),
            timeout=45,
        ) as response:
            archive = response.read(512 * 1024 * 1024)
        break
    except Exception as exc:
        last_error = exc
if archive is None:
    raise RuntimeError(f"Cannot download official RIFE 4.26 weights: {last_error}")

with zipfile.ZipFile(io.BytesIO(archive)) as zf:
    matches = [x for x in zf.infolist() if Path(x.filename).name == "flownet.pkl"]
    assert len(matches) == 1, "Missing flownet.pkl"
    with zf.open(matches[0]) as stream:
        state = torch.load(io.BytesIO(stream.read()), weights_only=True, map_location="cpu")
if "state_dict" in state:
    state = state["state_dict"]
state = {k.removeprefix("module."): v for k, v in state.items()}
model = module.IFNet(arch_ver="4.26")
model.load_state_dict(state, strict=True)
model.eval()
with torch.inference_mode():
    img = torch.rand(1, 3, 64, 64)
    output = model(
        img, img, torch.tensor([0.5]).reshape(1, 1, 1, 1),
        scale_list=[16, 8, 4, 2, 1],
        training=False, fastmode=True, ensemble=False,
    )
    assert output.shape == img.shape, output.shape
    assert torch.isfinite(output).all()
print("PASS: official flownet.pkl is strictly compatible with fork IFNet 4.26")
