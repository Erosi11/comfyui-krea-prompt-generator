# ComfyUI Krea Prompt Generator

Builds image prompts of women from dropdown selections and has your text encoder's LLM
(e.g. `qwen3vl_4b_fp8_scaled.safetensors` for Krea 2) write the final prompt. The node
outputs ready-to-use **CONDITIONING**, plus the prompt as a **STRING**.

## Requirements

- A recent ComfyUI build that includes the core **Generate Text** (`TextGenerate`) node.
  This custom node uses the same `clip.generate` API.
- A text encoder that can generate text, loaded with **CLIPLoader**. For Krea 2, use
  `qwen3vl_4b_fp8_scaled.safetensors` with type `krea2`.
- No extra Python packages.

## Installation

Copy (or `git clone`) this folder into `ComfyUI/custom_nodes/`, then restart ComfyUI.
The nodes appear under the **KreaPromptGenerator** category.

## Nodes

| Node | Purpose |
|------|---------|
| **Krea Prompt - Female Person** | Age, nationality, body, face, eyes, hair, skin, tattoos |
| **Krea Prompt - Female Fashion** | Outfit pieces with colors and materials, shoes, accessories, jewelry, nails |
| **Krea Prompt - Female Actions** | Standing, sitting, kneeling or lying pose; hands, legs, head; props |
| **Krea Prompt - Scene** | Style, location, time, season, weather, lighting, camera |
| **Krea Prompt Generator** | Takes CLIP, the attributes and your keywords; outputs `conditioning`, `prompt` and `llm_request` |
| **Krea Prompt - Save Unique Prompt** | Appends each new prompt to a text file, skipping prompts already in it, and shows the file on the node |

Every dropdown has three kinds of choice:

- **none**: the attribute is left out of the request sent to the LLM.
- **random**: the generator picks a value, driven by its `seed`. The same seed always gives the same picks.
- **a specific value**: passed through as-is.

Each attribute node also has an optional free-text box for extra details in that category.

## Wiring

```
[Female Person] -> [Female Fashion] -> [Female Actions] -> [Scene] --attributes--+
                                                                                 v
CLIPLoader (krea2) --clip------------------------------------> [Krea Prompt Generator]
                                                                   |-> conditioning -> KSampler positive
                                                                   |-> prompt       -> Preview Any (optional)
                                                                   |-> llm_request  (what was sent to the LLM)
```

You can use any subset of the attribute nodes, in any order. The generator also works
with no attributes connected, using only the keywords.

A starter graph is in `example_workflows/krea_prompt_generator.json`.

## Generator settings

| Setting | Meaning |
|---------|---------|
| `keywords` | Optional keywords or phrases that must appear in the prompt. Supports `{a\|b}` dynamic prompts. |
| `seed` | Drives the random picks and the LLM sampling. Set it to *randomize* for a new prompt on every run. |
| `min_words` / `max_words` | Target length of the written prompt |
| `max_length` | Maximum number of tokens the LLM may generate |
| `temperature`, `top_k`, `top_p`, `min_p`, `repetition_penalty` | LLM sampling, with the same defaults as the core Generate Text node. A temperature of 0 means greedy decoding. |
| `image`, `mask`, `reference_use` (optional) | Reference image; see below |
| `instructions` (optional input) | Replaces the built-in instructions. `{min_words}` and `{max_words}` are filled in for you. |

## Reference image

The generator has optional `image` and `mask` inputs, so the LLM can look at a reference while it
writes the prompt. This needs a vision-capable text encoder; Qwen3-VL qualifies.

- `reference_use` sets what the reference is used for: `overall look`, `subject appearance`, `outfit`,
  `pose`, `setting`, or `style & lighting`.
- Your dropdown selections and keywords always win over the reference when they conflict.
- `mask` focuses the reference on one area, using the area you paint in ComfyUI's mask editor. The
  image is cropped to the masked area, and the unmasked parts of that crop are filled with gray. An
  empty mask is ignored.
- Images are scaled down to 1024 px on the longer side. Up to 4 images from a batch are used.
- The reference only affects the written prompt. The conditioning is encoded from that text.

## Saving prompts

Connect the generator's `prompt` output to **Save Unique Prompt** and set `file_path`.

- Enter just a folder and file name, such as `kreaprompts\prompts1.txt`, and it's saved inside the ComfyUI `output` folder. A leading `\` is fine, and `.txt` is added if you leave the extension off. A full path with a drive letter, such as `D:\prompts\krea.txt`, saves there instead.
- The file and any missing folders are created on first use.
- Prompts are separated by a blank line. A prompt is skipped if the file already contains it, ignoring differences in spacing.
- The node shows the file path, the prompt count, whether the latest prompt was added, and the newest 100 prompts.
- You can edit the file by hand. Keep a blank line between prompts, because that's how entries are told apart.

## Credits

The attribute lists in `data/attributes.json` come from
[comfyui-character-prompt-builder](https://github.com/euan-gwd/comfyui-character-prompt-builder)
(MIT). Text generation follows ComfyUI core's `TextGenerate` node.
