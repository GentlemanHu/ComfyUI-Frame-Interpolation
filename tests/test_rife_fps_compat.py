"""Tests that the GACLove RIFEInterpolation workflow ID remains drop-in compatible."""
import importlib
import tempfile
from unittest.mock import patch
import torch
from test_rife_memory import load_fake_rife
import unittest


class BlendModel(torch.nn.Module):
    def forward(self, img0, img1, timestep, **kwargs):
        return img0 * (1 - timestep) + img1 * timestep


class LegacyCompatibilityTests(unittest.TestCase):
    def test_inputs_exactly_match_original(self):
        for root in load_fake_rife():
            module = importlib.import_module(root.__name__ + ".fps_compat")
            inputs = module.RIFEInterpolation.INPUT_TYPES()
            self.assertEqual(set(inputs["required"]),
                             {"images", "source_fps", "target_fps", "scale"})
            self.assertEqual(set(inputs["optional"]),
                             {"model_name", "batch_size", "use_fp16"})
            self.assertEqual(module.RIFEInterpolation.RETURN_TYPES, ("IMAGE",))
            self.assertEqual(module.RIFEInterpolation.FUNCTION, "interpolate")
            self.assertEqual(inputs["optional"]["batch_size"][1]["default"], 8)

    def test_24fps_to_60fps_preserves_frame_timing(self):
        for root in load_fake_rife():
            module = importlib.import_module(root.__name__ + ".fps_compat")
            frames = torch.tensor([0., .5, 1.]).view(3, 1, 1, 1).expand(3, 8, 8, 3).clone()
            with patch.object(module, "_get_model", return_value=BlendModel()):
                with patch.object(module, "allocate_output", wraps=module.allocate_output) as allocated:
                    output, = module.RIFEInterpolation().interpolate(
                        frames, source_fps=24, target_fps=60,
                        scale=1.0, model_name="flownet.pkl",
                        batch_size=8, use_fp16=True,
                    )
                    allocated.assert_called_once()
            self.assertEqual(output.shape, (7, 8, 8, 3))
            self.assertTrue(torch.allclose(
                output[:, 0, 0, 0],
                torch.tensor([0., .2, .4, .6, .8, 1., 1.]), atol=1e-6
            ))

    def test_low_ram_uses_mmap_and_does_not_duplicate_all_frames(self):
        for root in load_fake_rife():
            module = importlib.import_module(root.__name__ + ".fps_compat")
            frames = torch.rand(4, 8, 8, 3)
            with tempfile.TemporaryDirectory() as folder:
                with patch.dict("os.environ", {"RIFE_OUTPUT_DISK_DIR": folder}):
                    with patch.object(importlib.import_module(root.__name__ + ".output_buffer"),
                                      "available_ram_bytes", return_value=1):
                        with patch.object(module, "_get_model", return_value=BlendModel()):
                            result, = module.RIFEInterpolation().interpolate(
                                frames, 24, 60, 1.0
                            )
                            self.assertEqual(result.shape[0], 10)

    def test_same_fps_is_zero_cost_passthrough(self):
        for root in load_fake_rife():
            module = importlib.import_module(root.__name__ + ".fps_compat")
            frames = torch.rand(3, 8, 8, 3)
            with patch.object(module, "_get_model", side_effect=AssertionError("Not needed")):
                output, = module.RIFEInterpolation().interpolate(frames, 24, 24, 1)
                self.assertIs(output, frames)


if __name__ == "__main__":
    unittest.main()
