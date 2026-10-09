"""
ComfyUI Krea Prompt Generator
Builds image prompts for women from selectable attributes, written by the
text encoder's own LLM (e.g. Qwen3-VL 4B for Krea 2) and output as CONDITIONING.
"""

from .attributes import NODE_CLASS_MAPPINGS as ATTRIBUTE_CLASS_MAPPINGS
from .attributes import NODE_DISPLAY_NAME_MAPPINGS as ATTRIBUTE_DISPLAY_NAME_MAPPINGS
from .generator import NODE_CLASS_MAPPINGS as GENERATOR_CLASS_MAPPINGS
from .generator import NODE_DISPLAY_NAME_MAPPINGS as GENERATOR_DISPLAY_NAME_MAPPINGS
from .save_prompt import NODE_CLASS_MAPPINGS as SAVE_CLASS_MAPPINGS
from .save_prompt import NODE_DISPLAY_NAME_MAPPINGS as SAVE_DISPLAY_NAME_MAPPINGS

__version__ = "0.1.0"

NODE_CLASS_MAPPINGS = {**ATTRIBUTE_CLASS_MAPPINGS, **GENERATOR_CLASS_MAPPINGS, **SAVE_CLASS_MAPPINGS}
NODE_DISPLAY_NAME_MAPPINGS = {
    **ATTRIBUTE_DISPLAY_NAME_MAPPINGS,
    **GENERATOR_DISPLAY_NAME_MAPPINGS,
    **SAVE_DISPLAY_NAME_MAPPINGS,
}

# Frontend script that shows the saved prompt file on the Save Unique Prompt node
WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
