"""
Save Unique Prompt node.

Appends a prompt to a text file only if that file doesn't already contain it,
with a blank line between prompts, and shows the file's contents on the node.
"""

import os
import re

from .attributes import CATEGORY

try:
    import folder_paths

    BASE_DIR = folder_paths.get_output_directory()
except ImportError:  # running outside ComfyUI (tests)
    BASE_DIR = os.getcwd()

# Only the newest prompts are sent to the node display so huge files don't bog down the UI.
DISPLAY_LIMIT = 100

BLANK_LINE = re.compile(r"\n[ \t]*\n")


def resolve_path(file_path):
    path = os.path.expandvars(os.path.expanduser(file_path.strip().strip('"')))
    if not os.path.isabs(path):
        path = os.path.join(BASE_DIR, path)
    return os.path.normpath(path)


def normalize(text):
    return " ".join(text.split())


def read_prompts(path):
    if not os.path.exists(path):
        return "", []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    return text, [p.strip() for p in BLANK_LINE.split(text) if p.strip()]


class KreaSaveUniquePrompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"forceInput": True}),
                "file_path": (
                    "STRING",
                    {
                        "default": "krea_prompts/prompts.txt",
                        "tooltip": "Text file to save prompts to. Relative paths are inside the ComfyUI output folder. "
                                   "Created if it doesn't exist.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY
    DESCRIPTION = "Adds the prompt to a text file if it isn't already in it, and shows the file on the node."

    def save(self, prompt, file_path):
        if not file_path or not file_path.strip():
            raise ValueError("Save Unique Prompt: file_path is empty.")
        path = resolve_path(file_path)

        # Blank lines separate prompts in the file, so a prompt can't contain one itself.
        entry = BLANK_LINE.sub("\n", (prompt or "").strip())
        text, prompts = read_prompts(path)

        status = "empty prompt, nothing saved"
        if entry:
            if normalize(entry) in {normalize(p) for p in prompts}:
                status = "already in file, not added"
            else:
                os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
                trailing_newlines = len(text) - len(text.rstrip("\n"))
                separator = "\n" * max(0, 2 - trailing_newlines) if text.strip() else ""
                with open(path, "a", encoding="utf-8") as f:
                    f.write(separator + entry + "\n")
                prompts.append(entry)
                status = "added"

        shown = prompts[-DISPLAY_LIMIT:]
        header = f"{path}\n{len(prompts)} prompt(s) saved. Latest: {status}."
        if len(prompts) > len(shown):
            header += f" Showing the newest {len(shown)}."
        display = header + "\n\n" + "\n\n".join(shown)

        return {"ui": {"krea_file_text": [display]}, "result": (prompt,)}


NODE_CLASS_MAPPINGS = {"KreaSaveUniquePrompt": KreaSaveUniquePrompt}
NODE_DISPLAY_NAME_MAPPINGS = {"KreaSaveUniquePrompt": "Krea Prompt - Save Unique Prompt"}
