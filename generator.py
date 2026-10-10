"""
Krea Prompt Generator node.

Collects the chained attributes and the user's keywords, asks the CLIP model's
LLM (e.g. qwen3vl_4b loaded with CLIPLoader type "krea2") to write an image
prompt, then encodes that prompt into CONDITIONING exactly like CLIPTextEncode.

Text generation follows ComfyUI core's TextGenerate node
(comfy_extras/nodes_textgen.py): clip.tokenize -> clip.generate -> clip.decode.
"""

import inspect
import random
import re

from .attributes import ATTRIBUTES_TYPE, CATEGORY, RANDOM

DEFAULT_INSTRUCTIONS = """You are an expert prompt engineer for text-to-image models. Write a prompt for a text-to-image model that depicts one adult woman, based on the selections listed below. Output one paragraph of {min_words}-{max_words} words in plain descriptive sentences, not a comma-separated tag list.

Cover these in order, skipping any that don't apply:
1. Medium and style (candid phone photo, studio fashion shot, film photograph, oil painting, 3D render, etc.)
2. The woman: age, ethnicity, build and body, face, eyes, hair, skin, expression
3. Clothing, footwear and accessories, with exact colors and materials
4. Pose, body position, hands, and anything she is holding or doing
5. Setting and background
6. Lighting: direction, softness, color temperature, time of day
7. Camera: framing, angle, distance, lens, depth of field

Rules:
- Include every listed selection faithfully; rephrase them into natural, flowing language rather than copying labels.
- Every "Additional details" and "User keywords" item must appear in the prompt.
- For aspects that are not listed, add only small, plausible details that make the image coherent. Do not contradict any listed selection.
- If two selections conflict (for example two different poses, or a dress and separate pants), keep the one listed first and drop the other.
- Use specific colors ("rust orange", "pale sage") instead of generic ones.
- Do not begin with "The image shows" or "This is". Do not add quality words like masterpiece, 8k, or highly detailed.
- Output only the prompt, with no heading, preamble, or notes."""


# Generations that echo the instructions are retried with the next seed.
MAX_ATTEMPTS = 3

# Reference images are downscaled to this longer side before going to the vision encoder (VRAM / speed).
REFERENCE_MAX_SIDE = 1024
REFERENCE_MAX_IMAGES = 4
VISION_BLOCK = "<|vision_start|><|image_pad|><|vision_end|>"

# What the reference image should be used for.
REFERENCE_USES = {
    "overall look": "everything the selections don't specify: medium and style, the woman's appearance, clothing, "
                    "pose, setting, lighting and camera",
    "subject appearance": "the woman's physical appearance only (face, hair, body, skin)",
    "outfit": "clothing, footwear and accessories only",
    "pose": "pose, body position and gesture only",
    "setting": "setting, location and background only",
    "style & lighting": "medium, artistic style, color palette, lighting and camera only",
}


def prepare_reference(image, mask=None):
    """Return a list of single-image tensors [1, H, W, C] ready for the vision encoder, and whether a mask was applied.

    The mask (1 = area of interest, as painted in ComfyUI's mask editor) crops each image to the
    masked region and fills the rest of that crop with neutral gray. An empty mask is ignored.
    """
    import torch
    import torch.nn.functional as F

    images = image[:REFERENCE_MAX_IMAGES].float()
    masked = False
    out = []
    for i in range(images.shape[0]):
        img = images[i:i + 1, ..., :3]  # [1, H, W, 3], drop alpha if any
        h, w = img.shape[1], img.shape[2]

        if mask is not None and mask.numel() > 0:
            m = mask if mask.dim() == 3 else mask.unsqueeze(0)
            m = m[min(i, m.shape[0] - 1)].float().to(img.device)
            if m.shape != (h, w):
                m = F.interpolate(m[None, None], size=(h, w), mode="bilinear", align_corners=False)[0, 0]
            m = m.clamp(0, 1)
            ys, xs = torch.nonzero(m > 0.5, as_tuple=True)
            if len(ys) > 0:
                masked = True
                img = img * m[None, :, :, None] + 0.5 * (1 - m[None, :, :, None])
                pad = int(0.05 * max(h, w))
                y0, y1 = max(int(ys.min()) - pad, 0), min(int(ys.max()) + pad + 1, h)
                x0, x1 = max(int(xs.min()) - pad, 0), min(int(xs.max()) + pad + 1, w)
                img = img[:, y0:y1, x0:x1, :]

        h, w = img.shape[1], img.shape[2]
        scale = REFERENCE_MAX_SIDE / max(h, w)
        if scale < 1:
            size = (max(int(round(h * scale)), 1), max(int(round(w * scale)), 1))
            img = F.interpolate(img.movedim(-1, 1), size=size, mode="area").movedim(1, -1)
        out.append(img.clamp(0, 1))
    return out, masked


def reference_instructions(use, masked, count):
    what = "image is" if count == 1 else f"{count} images are"
    text = (f"Reference: {what} attached. Use the reference for {REFERENCE_USES.get(use, REFERENCE_USES['overall look'])}. "
            "Describe what you take from it in your own words. The listed selections and user keywords always take priority; "
            "where they conflict with the reference, follow the selections. Never mention the reference image itself.")
    if masked:
        text += " Only the area of interest is shown; flat gray regions are masking, not part of the scene."
    return text


def resolve_attributes(entries, seed):
    """Replace "random" values with a seeded choice; returns (group, label, value) tuples.

    Each field gets its own RNG so changing one dropdown doesn't reshuffle the others.
    """
    resolved = []
    for entry in entries or []:
        value = entry["value"]
        if value == RANDOM:
            options = entry.get("options") or []
            if not options:
                continue
            value = random.Random(f"{seed}:{entry['field']}").choice(options)
        resolved.append((entry["group"], entry["label"], value))
    return resolved


def build_system_prompt(instructions, min_words, max_words):
    return instructions.replace("{min_words}", str(min_words)).replace("{max_words}", str(max_words)).strip()


def build_user_message(resolved, keywords, has_reference=False):
    lines = ["Subject: one adult woman."]

    groups = []
    for group, _label, _value in resolved:
        if group not in groups:
            groups.append(group)
    for group in groups:
        lines.append("")
        lines.append(f"{group}:")
        for g, label, value in resolved:
            if g == group:
                lines.append(f"- {label}: {value}")

    keywords = (keywords or "").strip()
    if keywords:
        lines.append("")
        lines.append("User keywords (must be included):")
        lines.append(keywords)

    if not resolved and not keywords:
        lines.append("")
        lines.append("No selections were made; base the prompt on the reference." if has_reference
                     else "No selections were made; invent a tasteful, coherent portrait.")

    lines.append("")
    lines.append("Write the image prompt now.")
    return "\n".join(lines)


def chatml(system_prompt, user_message, image_count=0):
    """Full Qwen chat turn with our instructions as the real system prompt.

    Without this, the krea2 tokenizer wraps text-only requests in its conditioning template,
    whose system prompt ("Describe the image by detailing...") conflicts with ours and makes
    the model sometimes repeat instructions instead of writing a prompt. The empty think
    block is the Qwen3 convention for "no reasoning".
    """
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{VISION_BLOCK * image_count}{user_message}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


ECHO_MARKERS = (
    "describe the image by detailing",
    "you are an expert prompt engineer",
    "cover these in order",
    "subject: one adult woman",
    "user keywords",
    "write the image prompt now",
    "reference image",
    "im_start",
    "im_end",
)


def echo_reason(text, system_prompt):
    """Why the output looks like repeated instructions/selections instead of a prompt, or None if it looks fine."""
    if not text:
        return "empty output"
    low = text.lower()
    for marker in ECHO_MARKERS:
        if marker in low:
            return f'contains "{marker}"'
    for line in system_prompt.splitlines():
        line = line.strip().lstrip("-0123456789. ").lower()
        if len(line) > 30 and line in low:
            return f'repeats instruction "{line[:60]}"'
    # Two or more "- Label: value" lines means the selection list was copied.
    if len(re.findall(r"^\s*-\s*[^:\n]{1,40}:", text, re.MULTILINE)) >= 2:
        return 'contains a "- Label: value" list'
    return None


def looks_like_echo(text, system_prompt):
    return echo_reason(text, system_prompt) is not None


def reset_generation_state():
    """Drop ComfyUI's cached CUDA graphs and cast buffers between generate() calls.

    ComfyUI records CUDA graphs for the decode step on the model's layers and only clears them after
    a node finishes (execution.py -> cleanup_prefetch_queues). A second generate() in the same node
    would replay graphs recorded against the previous call's freed KV cache, causing a CUDA
    "scatter gather kernel index out of bounds" crash. This mirrors what execution.py does between nodes.
    """
    try:
        import comfy.memory_management
        import comfy.model_management
        import comfy.model_prefetch
    except ImportError:
        return  # older ComfyUI without the model compiler: nothing to reset
    if not getattr(comfy.memory_management, "aimdo_enabled", False):
        return
    comfy.model_prefetch.cleanup_prefetch_queues()
    comfy.model_management.reset_cast_buffers()


def call_supported(fn, *args, **kwargs):
    """Call fn with only the keyword arguments its signature accepts (older ComfyUI builds lack some)."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return fn(*args, **kwargs)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return fn(*args, **kwargs)
    return fn(*args, **{k: v for k, v in kwargs.items() if k in params})


def strip_reasoning(text):
    reasoning, sep, rest = text.partition("</think>")
    if sep and reasoning.lstrip().startswith("<think>"):
        text = rest
    return text.strip().strip('"').strip()


class KreaPromptGenerator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP", {"tooltip": "Text encoder, e.g. qwen3vl_4b_fp8_scaled loaded with CLIPLoader type krea2"}),
                "keywords": (
                    "STRING",
                    {
                        "multiline": True,
                        "dynamicPrompts": True,
                        "default": "",
                        "placeholder": "Optional keywords, phrases or details to include in the prompt",
                    },
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True,
                                 "tooltip": "Drives both the 'random' attribute picks and the LLM sampling"}),
                "min_words": ("INT", {"default": 80, "min": 10, "max": 1000}),
                "max_words": ("INT", {"default": 200, "min": 10, "max": 1000}),
                "max_length": ("INT", {"default": 512, "min": 16, "max": 4096, "tooltip": "Maximum tokens the LLM may generate"}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.01,
                                          "tooltip": "0 = greedy (deterministic, no sampling)"}),
                "top_k": ("INT", {"default": 64, "min": 0, "max": 1000}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
                "min_p": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
                "repetition_penalty": ("FLOAT", {"default": 1.05, "min": 0.0, "max": 5.0, "step": 0.01}),
            },
            "optional": {
                "attributes": (ATTRIBUTES_TYPE, {"tooltip": "Output of the last Krea Prompt attribute node"}),
                "instructions": ("STRING", {"forceInput": True,
                                            "tooltip": "Replaces the built-in instructions sent to the LLM. "
                                                       "{min_words} and {max_words} are substituted."}),
                "image": ("IMAGE", {"tooltip": "Optional reference image (up to 4 from a batch)"}),
                "mask": ("MASK", {"tooltip": "Optional: focus the reference on the masked area"}),
                "reference_use": (list(REFERENCE_USES), {"default": "overall look",
                                                         "tooltip": "What the reference image should be used for"}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "STRING", "STRING")
    RETURN_NAMES = ("conditioning", "prompt", "llm_request")
    FUNCTION = "generate"
    CATEGORY = CATEGORY
    DESCRIPTION = "Writes an image prompt with the text encoder's LLM from the selected attributes, then encodes it."

    def generate(self, clip, keywords, seed, min_words, max_words, max_length, temperature, top_k, top_p,
                 min_p, repetition_penalty, attributes=None, instructions=None, image=None, mask=None,
                 reference_use="overall look"):
        if clip is None:
            raise RuntimeError("Krea Prompt Generator: no CLIP model connected.")
        if not hasattr(clip, "generate"):
            raise RuntimeError(
                "Krea Prompt Generator: this ComfyUI build cannot generate text with CLIP models. "
                "Update ComfyUI to a version that includes the core 'Generate Text' (TextGenerate) node."
            )

        min_words, max_words = min(min_words, max_words), max(min_words, max_words)
        resolved = resolve_attributes(attributes, seed)
        system_prompt = build_system_prompt(instructions or DEFAULT_INSTRUCTIONS, min_words, max_words)
        references, masked = prepare_reference(image, mask) if image is not None else ([], False)
        if references:
            system_prompt += "\n\n" + reference_instructions(reference_use, masked, len(references))
        user_message = build_user_message(resolved, keywords, has_reference=bool(references))
        request = f"[SYSTEM]\n{system_prompt}\n\n[USER]\n"
        if references:
            request += f"<{len(references)} reference image(s){', masked' if masked else ''}>\n"
        request += user_message

        # Qwen tokenizers (krea2, qwen3vl) use ChatML: build the whole turn ourselves so our
        # instructions replace the conditioning template's system prompt. Each vision block's
        # <|image_pad|> is swapped for one of the passed images by the tokenizer.
        template = getattr(getattr(clip, "tokenizer", None), "llama_template", None) or ""
        if "<|im_start|>" in template:
            tokens = call_supported(clip.tokenize, chatml(system_prompt, user_message, len(references)),
                                    images=references, skip_template=True, min_length=1)
        else:
            tokens = call_supported(clip.tokenize, f"{system_prompt}\n\n{user_message}", image=image,
                                    skip_template=False, min_length=1, thinking=False)

        prompt = ""
        for attempt in range(MAX_ATTEMPTS):
            if attempt > 0:
                reset_generation_state()
            generated_ids = call_supported(
                clip.generate,
                tokens,
                do_sample=temperature > 0,
                max_length=max_length,
                temperature=max(temperature, 0.01),
                top_k=top_k,
                top_p=top_p,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
                presence_penalty=0.0,
                seed=(seed + attempt) % 0x10000000000000000,
                mtp=True,
            )
            prompt = strip_reasoning(clip.decode(generated_ids))
            reason = echo_reason(prompt, system_prompt)
            if reason is None:
                break
            preview = " ".join(prompt.split())[:400]
            print(f"[KreaPromptGenerator] Attempt {attempt + 1}/{MAX_ATTEMPTS} rejected ({reason}). Output was: {preview}")
            if temperature <= 0:
                break  # greedy decoding would just repeat the same output
        if prompt and looks_like_echo(prompt, system_prompt):
            print("[KreaPromptGenerator] Warning: could not get a clean prompt; using the last attempt.")

        if not prompt:
            raise RuntimeError("Krea Prompt Generator: the LLM returned an empty prompt. Try another seed or a larger max_length.")

        # Encode like CLIPTextEncode so the output plugs straight into the sampler. Core ComfyUI does
        # generation and encoding in separate nodes with a graph cleanup in between; do the same.
        reset_generation_state()
        conditioning = clip.encode_from_tokens_scheduled(clip.tokenize(prompt))
        return (conditioning, prompt, request)


NODE_CLASS_MAPPINGS = {"KreaPromptGenerator": KreaPromptGenerator}
NODE_DISPLAY_NAME_MAPPINGS = {"KreaPromptGenerator": "Krea Prompt Generator"}
