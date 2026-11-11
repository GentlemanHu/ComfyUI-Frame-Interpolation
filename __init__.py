import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from .other_nodes import Gradually_More_Denoise_KSampler

#Some models are commented out because the code is not completed
#from vfi_models.eisai import EISAI_VFI
from vfi_models.gmfss_fortuna import GMFSS_Fortuna_VFI
from vfi_models.ifrnet import IFRNet_VFI
from vfi_models.ifunet import IFUnet_VFI
from vfi_models.m2m import M2M_VFI
from vfi_models.rife import RIFE_VFI
from vfi_models.rife.fps_compat import RIFEInterpolation
from vfi_models.sepconv import SepconvVFI
from vfi_models.amt import AMT_VFI
from vfi_models.film import FILM_VFI
from vfi_models.stmfnet import STMFNet_VFI
from vfi_models.flavr import FLAVR_VFI
from vfi_models.cain import CAIN_VFI
from vfi_models.atm import ATM_VFI
from vfi_models.momo import MOMO_VFI
from vfi_utils import MakeInterpolationStateList, FloatToInt
    
NODE_CLASS_MAPPINGS = {
    "KSampler Gradually Adding More Denoise (efficient)": Gradually_More_Denoise_KSampler,
#    "EISAI VFI": EISAI_VFI,
    "GMFSS Fortuna VFI": GMFSS_Fortuna_VFI,
    "IFRNet VFI": IFRNet_VFI,
    "IFUnet VFI": IFUnet_VFI,
    "M2M VFI": M2M_VFI,
    "RIFE VFI": RIFE_VFI,
    "RIFEInterpolation": RIFEInterpolation,
    "Sepconv VFI": SepconvVFI,
    "AMT VFI": AMT_VFI,
    "FILM VFI": FILM_VFI,
    "Make Interpolation State List": MakeInterpolationStateList,
    "STMFNet VFI": STMFNet_VFI,
    "FLAVR VFI": FLAVR_VFI,
    "CAIN VFI": CAIN_VFI,
    "ATM VFI": ATM_VFI,
    "MOMO VFI": MOMO_VFI,
    "VFI FloatToInt": FloatToInt
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RIFE VFI": "RIFE VFI (recommend rife4.25+)",
    "RIFEInterpolation": "RIFE Frame Interpolation (RAM Safe)",
    "ATM VFI": "ATM VFI (only supports 2x multiplier)",
    "MOMO VFI": "MOMO VFI (only supports 2x multiplier)"
}


def _register_legacy_rife_precedence():
    """Preserve legacy RIFEInterpolation behavior with both plugins installed.

    ComfyUI loads custom nodes in filesystem order; an older GACLove package
    may register the same node ID after this fork. Restore our implementation
    immediately before prompt validation, without modifying the other plugin.
    """
    try:
        from server import PromptServer
        import nodes
        import logging
        server = PromptServer.instance
        if server is None:
            logging.warning("RIFE compatibility: PromptServer unavailable at import")
            return

        def _prefer_safe_rife(prompt_json):
            prompt = prompt_json.get("prompt", {}) if isinstance(prompt_json, dict) else {}
            if isinstance(prompt, dict) and any(
                isinstance(v, dict) and v.get("class_type") == "RIFEInterpolation"
                for v in prompt.values()
            ):
                if nodes.NODE_CLASS_MAPPINGS.get("RIFEInterpolation") is not RIFEInterpolation:
                    nodes.NODE_CLASS_MAPPINGS["RIFEInterpolation"] = RIFEInterpolation
                    nodes.NODE_DISPLAY_NAME_MAPPINGS["RIFEInterpolation"] = (
                        "RIFE Frame Interpolation (RAM Safe)"
                    )
                    logging.info("RIFE compatibility: activated RAM-safe RIFEInterpolation")
            return prompt_json

        server.add_on_prompt_handler(_prefer_safe_rife)
    except (ImportError, AttributeError) as exc:
        import logging
        logging.warning("RIFE compatibility: cannot install node precedence hook: %s", exc)


_register_legacy_rife_precedence()
