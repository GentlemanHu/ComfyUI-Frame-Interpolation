"""Verify unchanged GACLove node ID wins at prompt time even if loaded last."""
import ast
import logging
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


class NodePrecedenceTests(unittest.TestCase):
    def test_prompts_restore_fork_after_later_duplicate_registration(self):
        module = ast.parse((ROOT / "__init__.py").read_text(encoding="utf-8"))
        function = next(
            node for node in module.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_register_legacy_rife_precedence"
        )
        scripted = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
        namespace = {"RIFEInterpolation": type("SafeRIFE", (), {})}
        exec(compile(scripted, str(ROOT / "__init__.py"), "exec"), namespace)
        hooks = []
        server = types.ModuleType("server")
        server.PromptServer = type(
            "PromptServer", (), {"instance": type(
                "Instance", (), {"add_on_prompt_handler": lambda self, cb: hooks.append(cb)}
            )()}
        )
        nodes = types.ModuleType("nodes")
        gac_cls = type("GACLegacyRIFE", (), {})
        nodes.NODE_CLASS_MAPPINGS = {"RIFEInterpolation": gac_cls}
        nodes.NODE_DISPLAY_NAME_MAPPINGS = {"RIFEInterpolation": "RIFE"}
        with patch.dict(sys.modules, {"server": server, "nodes": nodes}):
            namespace["_register_legacy_rife_precedence"]()
            self.assertEqual(len(hooks), 1)
            other = {"prompt": {"1": {"class_type": "SomeOtherNode"}}}
            self.assertIs(hooks[0](other), other)
            self.assertIs(nodes.NODE_CLASS_MAPPINGS["RIFEInterpolation"], gac_cls)
            original = {"prompt": {"7": {"class_type": "RIFEInterpolation"}}}
            self.assertIs(hooks[0](original), original)
            self.assertIs(
                nodes.NODE_CLASS_MAPPINGS["RIFEInterpolation"],
                namespace["RIFEInterpolation"],
            )


if __name__ == "__main__":
    unittest.main()
