"""CPU-only regression tests for memory buffering and RIFE inference orchestration."""
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]


def load_buffer():
    path = ROOT / "vfi_models" / "rife" / "output_buffer.py"
    spec = importlib.util.spec_from_file_location("rife_buffer_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BufferTests(unittest.TestCase):
    def test_frame_layout_with_skips(self):
        b = load_buffer()
        original, tasks, count = b.output_layout(4, [3, 9, 2], [False, True, False])
        self.assertEqual(original, [0, 3, 4])
        self.assertEqual(count, 7)
        self.assertEqual(tasks, [(0, 1/3, 1), (0, 2/3, 2), (2, .5, 5)])

    def test_disk_buffer_and_temp_cleanup(self):
        b = load_buffer()
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {"RIFE_OUTPUT_DISK_DIR": directory}):
                tensor = b.allocate_output((5, 3, 8, 8), storage="disk")
                tensor.fill_(0.25)
                self.assertAlmostEqual(tensor.mean().item(), 0.25)
                self.assertFalse(os.listdir(directory))

    def test_auto_switches_to_disk_on_low_ram(self):
        b = load_buffer()
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {"RIFE_OUTPUT_DISK_DIR": directory}):
                with patch.object(b, "available_ram_bytes", return_value=1):
                    tensor = b.allocate_output((5, 3, 8, 8), "auto", 0)
                    tensor.fill_(0.75)
                    self.assertAlmostEqual(tensor.mean().item(), 0.75)


class DummyNet(torch.nn.Module):
    def __init__(self, arch_ver):
        super().__init__()

    def forward(self, img0, img1, timestep, *_):
        return img0 * (1 - timestep) + img1 * timestep


def load_fake_rife():
    vfi = types.ModuleType("vfi_utils")
    vfi.load_file_from_github_release = lambda *args: "unused.pth"
    vfi.preprocess_frames = lambda x: x[..., :3].permute(0, 3, 1, 2)
    vfi.postprocess_frames = lambda x: x.permute(0, 2, 3, 1).cpu()
    vfi.InterpolationStateList = type("InterpolationStateList", (), {})
    comfy = types.ModuleType("comfy")
    mgmt = types.ModuleType("comfy.model_management")
    mgmt.get_torch_device = lambda: torch.device("cpu")
    mgmt.soft_empty_cache = lambda: None
    comfy.model_management = mgmt
    arch = types.ModuleType("rife_unit.rife_arch")
    arch.IFNet = DummyNet
    path = ROOT / "vfi_models" / "rife" / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        "rife_unit", path, submodule_search_locations=[str(path.parent)]
    )
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {
        "vfi_utils": vfi, "comfy": comfy, "comfy.model_management": mgmt,
        "rife_unit": module, "rife_unit.rife_arch": arch,
    }):
        with patch("torch.load", return_value={}):
            spec.loader.exec_module(module)
            yield module


class InferenceTests(unittest.TestCase):
    def test_batched_interpolation_order(self):
        frames = torch.tensor([0., .5, 1.]).view(3, 1, 1, 1).expand(3, 8, 8, 3).clone()
        for module in load_fake_rife():
            with patch("torch.load", return_value={}):
                result, = module.RIFE_VFI().vfi(
                    "rife49.pth", frames, multiplier=2, batch_size=3, output_storage="ram"
                )
            self.assertEqual(tuple(result.shape), (5, 8, 8, 3))
            self.assertTrue(torch.allclose(
                result[:, 0, 0, 0], torch.tensor([0., .25, .5, .75, 1.])
            ))

    def test_skip_and_model_cache_eviction(self):
        frames = torch.tensor([0., .5, 1.]).view(3, 1, 1, 1).expand(3, 8, 8, 3).clone()
        class SkipFirst:
            def is_frame_skipped(self, i):
                return i == 0
        for module in load_fake_rife():
            with patch("torch.load", return_value={}):
                node = module.RIFE_VFI()
                result, = node.vfi("rife49.pth", frames, multiplier=2,
                                    optional_interpolation_states=SkipFirst(), output_storage="ram")
                self.assertEqual(len(module._model_cache), 1)
                node.vfi("rife49.pth", frames, multiplier=2,
                         dtype="float16", output_storage="ram")
            self.assertEqual(len(module._model_cache), 1)
            self.assertTrue(torch.allclose(result[:, 0, 0, 0], torch.tensor([0., .5, .75, 1.])))


if __name__ == "__main__":
    unittest.main()
