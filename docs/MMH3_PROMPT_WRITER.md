# MMH3 Prompt Writer

Seven nodes that write MiniMax H3 prompts inside ComfyUI with **any LLM you choose**. The model follows an H3 guide (Ref2VA by default), a checker tests the draft against the guide's hard rules, and anything that fails goes back to the model to fix. The result is saved, so a prompt can be **written first and used later** in any workflow. That includes the Reference to Video node, or the `/mmh3` page by pasting.

Ready-made graphs are in [`workflow/MMH3 Prompt Builder.json`](../workflow/MMH3%20Prompt%20Builder.json) and [`workflow/MMH3 Prompt Loader.json`](../workflow/MMH3%20Prompt%20Loader.json).

```
MMH3 Model Select ─┐
MMH3 Prompt Spec ──┼─► MMH3 Prompt Writer ─► MMH3 Prompt Save ─► ComfyUI/user/h3_prompts/<name>.txt
Load Image ×1–4 ───┘        (draft → check → fix)                          │
                                                                 MMH3 Prompt Load ─► Reference to Video.prompt
```

| Class | Display name | Outputs |
|-------|--------------|---------|
| `MMH3ModelSelect` | MMH3 Model Select | `MMH3_LLM` |
| `MMH3PromptSpec` | MMH3 Prompt Spec (guide) | `STRING` |
| `MMH3PromptWriter` | MMH3 Prompt Writer | `STRING` prompt, `STRING` report, `MMH3_META` |
| `MMH3PromptRefine` | MMH3 Prompt Refine | `STRING` prompt, `STRING` report, `MMH3_META` |
| `MMH3PromptCheck` | MMH3 Prompt Check | `STRING` prompt, `STRING` report, `BOOLEAN` passed |
| `MMH3PromptSave` | MMH3 Prompt Save | `STRING` |
| `MMH3PromptLoad` | MMH3 Prompt Load | `STRING` prompt, `STRING` mode, `FLOAT` duration, `STRING` report |

All are in the category `dihan-nodes/mmh3` and live in `mmh3_prompt/`. They need no installs beyond what ComfyUI already has: the standard library, numpy and Pillow.

## Choosing the model: MMH3 Model Select

The dropdown combines the presets in `mmh3_prompt/models.json` with **every model installed in Ollama** (discovered at `ollama_base_url`).

| Provider | Covers |
|---|---|
| `ollama` | Local models. `keep_alive: 0` unloads the model right after writing, so it doesn't sit in the VRAM H3 needs. |
| `anthropic` | Claude (Messages API). |
| `openai_compat` | OpenAI, Gemini (OpenAI-compatible endpoint), OpenRouter, LM Studio, llama.cpp server, vLLM, Groq and others. |

- **API keys:** use an environment variable named by `api_key_env` (for example `ANTHROPIC_API_KEY`), or `mmh3_prompt/api_keys.json` (copy `api_keys.example.json`). That file is git-ignored. Keys are never stored in the workflow.
- **`model_override` / `base_url_override`:** use any model name or server, for example another machine's Ollama.
- **`vision: true`:** the model sees the reference images. For text-only models, describe them in `references`.
- **Model names:** the preset names are examples. Edit `models.json` and press R if a provider renames a model.

## The guide: MMH3 Prompt Spec

`(auto: match mode)` picks `specs/minimax_h3_ref2va_guide.txt` for Ref2VA and `specs/minimax_h3_base_guide.txt` for T2VA / I2VA / FL2VA / L2VA. Any `.txt` added to `mmh3_prompt/specs/` appears in the list, and `extra_rules` appends house rules.

The Ref2VA guide follows MiniMax's official full-reference format ([`references/ref-en.txt`](https://github.com/MiniMax-AI/MiniMax-H3/blob/main/skills/h3-prompt-writing/references/ref-en.txt)). It adds the render rules learned in practice: dialogue appears once, mouth movement is described, only the four amplitude/speed phrases are allowed, and the soundscape keeps an ambience floor.

**The sections**, in this order:
- `subject_definitions`
- `summary` with a `[reference generation]` prefix
- `retention_analysis` with `fully_preserved` and similar markers
- `detailed_description`: a style line, then `[Shot 1]`
- `overall_soundscape`
- `non_diegetic_music`

**Labels:**
- `<Subject N>` is a person, place or prop, citing the picture it comes from.
- `<Picture N>` gets its own line only when it is a concrete frame or composition anchor. An example is the carry-over frame from the previous clip, scoped to blocking only.
- `<Audio N>` is used for voice-timbre references.

There is no base-mode alignment line. That line makes H3 treat references as keyframes.

## Writing: MMH3 Prompt Writer

| Input | Notes |
|---|---|
| `mode` | `Ref2VA` (default), T2VA, I2VA, FL2VA, L2VA, or `auto` (base modes by picture count). |
| `duration` | Total seconds; 0 = not set. Needed for FL2VA / L2VA and for more than one shot. |
| `idea` | What happens: action, camera, mood, sound. |
| `references` | What each picture is, for example `Picture 1: the woman - identity, hair, outfit`. Naming pictures here is enough to write a prompt **before the images are wired**. The checker then allows exactly those picture numbers. |
| `picture_1`–`picture_4` | Optional images. They are **numbered by slot**, like the Reference to Video node's `ref_image` slots, so an empty slot 3 keeps slot 4 as `<Picture 4>`. |
| `dialogue`, `visible_text` | Exact lines and on-screen text, kept verbatim. |
| `audio_references` | For example `Audio 1: voice timbre reference for the woman`. |
| `max_fix_rounds` | How many times check failures go back to the model (default 2). |
| `strict` | Also send warnings back, not just errors. |
| `autofix` | Repairs section order, layout and stray alignment lines without calling the model. |
| `seed` | Change it for a new draft. The same seed with the same inputs is served from ComfyUI's cache without calling the model again. |

The report on the node (and the `report` output) shows PASS/FAIL, each broken rule and each round. The model is never allowed to stop and ask a question. If it tries, the run fails with that question so you can add the answer to the inputs.

## After a render: MMH3 Prompt Refine

Pick what went wrong and/or type feedback. It revises the prompt with the same model and guide, then re-checks it. Symptoms include:
- voice plays but the mouth stays still
- a line is spoken twice
- click at the start
- wrong camera move
- identity or room drift
- copied the reference too literally
- carry-over frame copied too literally or soft

The mode, duration and references come from `meta`.

## Check, Save, Load

- **Check:** runs the checker alone on any prompt, including hand-written ones and prompts assembled by the `/mmh3` page. No model needed. Set `reference_pictures` to check picture numbers.
- **Save:** writes `ComfyUI/user/h3_prompts/<name>.txt`, plus a `.json` with the mode, duration, model, pictures and report. With `overwrite` off, a name that already exists gets `_2`, `_3` and so on.
- **Load:** a dropdown of saved prompts (press R after saving), newest first. Hand edits to the `.txt` are picked up.

## What the checker enforces (Ref2VA)

**Errors:**
- **Sections:** all six are present, each once. Wrong order and stray alignment lines are auto-fixed. `integrated_multimodal_description` is never used.
- **subject_definitions:** every line starts with a label and "is". No label is defined twice.
- **Labels:** every label used anywhere is defined. The summary introduces no new labels. Picture numbers match the supplied references.
- **summary:** starts with valid task types (`reference generation`, `keyframe completion`, `video editing`, `video continuation`, `audio reuse`, `audio reference`), with no repeats.
- **retention_analysis:**
  - Lines use the form `<Label> (…): marker - reason`, with markers valid for their type.
  - There is one line per standalone label.
  - No `(S1)` appears here.
- **Shots:** a style line comes before `[Shot 1]`. `[Shot 1]` has no timestamp, later cut times increase and fall inside the duration, and there are no timestamps inside a shot.
- **Dialogue:**
  - Each line is spoken once. A fuzzy scan catches paraphrases in the description, the summary and the sound sections.
  - `<d>` blocks carry a language tag.
  - The mouth movement is described, and voiceovers use the exact phrase with lips closed.
  - Speakers are numbered by first vocal event.
  - Speaking shots contain no stillness wording.
  - The dialogue fits the clip at 3 words per second.
- **Camera:** no speed or amplitude wording beyond the four allowed phrases, and no capitalised motion keywords.
- **Soundscape:** no silence requests, recording notes, music or speech.
- **Wording:** no planning words and no citations.

**Warnings:**
- The description is outside roughly 350–500 words.
- A picture has its own line but is only a subject.
- A picture is connected but never cited.
- The task types don't match the defined frames or audio.
- The mouth doesn't close after a line.
- `non_diegetic_music: N/A` is used on a talking shot.
- Music is described with mood words.

**Base modes** get the same body rules, plus the exact alignment template for their mode (auto-fixed), no line breaks in `integrated_multimodal_description`, and a one-beat-per-second pacing check.

**Note about the `/mmh3` page's default blocks:** they put `retention_analysis` before `summary`, and the page's segment text has no `detailed_description:` header. MiniMax's documented order is subject_definitions → summary → retention_analysis → detailed_description. Run a page-assembled prompt through **MMH3 Prompt Check** to see the difference.

## Tests

`python3 tests/test_mmh3_prompt.py` runs offline. It covers:
- the checker on good, broken and messy prompts
- the three wire formats against a local mock server, including the fix-up round
- save and load
- both shipped workflows against `INPUT_TYPES`

After changing node inputs, regenerate the graphs with `python3 workflow/_build_mmh3_prompt.py`.
