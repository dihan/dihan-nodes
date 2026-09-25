"""MiniMax H3 prompt writer nodes: pick a model, give it the H3 guide, write -> check -> fix, save, load.

Ref2VA (full-reference mode) is the default; the base modes (T2VA / I2VA / FL2VA / L2VA)
are still available with their own guide.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import re

from . import providers
from .lint import BASE_MODES, MODES, REF_MODE, alignment_line, extract_prompt, infer_mode_and_duration, lint, \
    looks_like_question

SPECS_DIR = os.path.join(providers.PACK_DIR, "specs")
CATEGORY = "dihan-nodes/mmh3"
AUTO_SPEC = "(auto: match mode)"
SPEC_FOR_MODE = {REF_MODE: "minimax_h3_ref2va_guide.txt", **{m: "minimax_h3_base_guide.txt" for m in BASE_MODES}}
SPEC_SENTINEL = "\u0000mmh3-auto-spec\u0000"
MAX_PICTURES = 4

try:
    import folder_paths  # type: ignore

    SAVE_DIR = os.path.join(folder_paths.get_user_directory(), "h3_prompts")
except Exception:  # outside ComfyUI (tests)
    SAVE_DIR = os.environ.get("H3_PROMPTS_DIR", os.path.join(providers.PACK_DIR, "saved_prompts"))

try:
    from aiohttp import web
    from server import PromptServer
except Exception:  # outside ComfyUI (tests)
    web = PromptServer = None

NONE_SAVED = "(no saved prompts yet)"


# ====================================================================== helpers

def _read_spec(name: str) -> str:
    with open(os.path.join(SPECS_DIR, name), "r", encoding="utf-8") as f:
        return f.read().strip()


def _resolve_spec(spec: str, mode: str) -> str:
    """The Spec node sends a sentinel in auto mode; swap in the guide that fits the mode."""
    if spec.startswith(SPEC_SENTINEL):
        return _read_spec(SPEC_FOR_MODE[mode]) + spec[len(SPEC_SENTINEL):]
    return spec


def _resolve_mode(mode: str, n_images: int) -> str:
    if mode in MODES:
        return mode
    return {0: "T2VA", 1: "I2VA"}.get(n_images, "FL2VA")


def _check_duration(mode: str, duration: float):
    if mode in ("FL2VA", "L2VA") and not duration:
        raise ValueError(f"{mode} needs the exact total duration in seconds (the last picture must land on it). Set 'duration' and queue again.")


def _slots(*imgs):
    """[(picture number, image)] for the connected slots, numbered by slot like the Reference to Video node."""
    return [(i + 1, im) for i, im in enumerate(imgs) if im is not None]


def _pictures_named(text: str) -> list[int]:
    return sorted({int(n) for n in re.findall(r"\bPicture\s*(\d+)\b", text or "", re.I)})


def _context(mode, duration, slots, vision, references="", audio_refs=""):
    lines = [
        "PIPELINE CONTEXT: you are running inside an automated ComfyUI node and a script parses your reply.",
        "- Reply with the finished prompt only. A single code block is fine; nothing before or after it.",
        "- Do not ask questions. Everything needed is below; make sensible choices for small missing details.",
    ]
    nums = [n for n, _ in slots]
    if mode == REF_MODE:
        lines.append("- Mode: Ref2VA (full-reference mode). Use the six-section format: subject_definitions, summary, "
                     "retention_analysis, detailed_description, overall_soundscape, non_diegetic_music.")
        named = sorted(set(nums) | set(_pictures_named(references)))
        if nums:
            lines.append("- Reference images attached, in order: " + ", ".join(f"<Picture {n}>" for n in nums) + ".")
        if named:
            lines.append("- The ONLY picture labels that exist are: " + ", ".join(f"<Picture {n}>" for n in named) + ". Never use another number.")
        lines.append("- Do NOT write a base-mode alignment line and do NOT use integrated_multimodal_description.")
    else:
        lines.append(f"- Mode: {mode}.")
        lines.append({
            "T2VA": "- No reference picture. Do not mention Picture 1 or Picture 2.",
            "I2VA": "- <Picture 1> is the FIRST frame (the attached image).",
            "FL2VA": "- Picture 1 (first attached image) is the FIRST frame; Picture 2 (second attached image) is the LAST frame.",
            "L2VA": "- <Picture 1> (the attached image) is the LAST frame.",
        }[mode])
    if slots and not vision:
        lines.append("- The pictures are NOT visible to you. Rely on the reference notes below.")
    if duration:
        lines.append(f"- Total duration: {duration:.2f} seconds. Use exactly this.")
    elif mode == REF_MODE:
        lines.append("- No duration given: write one continuous [Shot 1] with no timestamps; keep the actions to what fits about 6-8 seconds.")
    else:
        lines.append("- No duration given: write one continuous [Shot 1] with no timestamps, and keep it lean (about 4-6 seconds of action).")
    if mode in BASE_MODES:
        al = alignment_line(mode, duration, 1) if (mode == "I2VA" or duration) else None
        if al:
            lines.append(f"- The first line must be exactly: {al}" + ("" if mode == "I2VA" else "  (change the shot number only if the video has more than one shot)"))
    out = "\n".join(lines)
    out += _section("WHAT EACH REFERENCE IS", references)
    out += _section("AUDIO REFERENCES (use <Audio N> labels)", audio_refs)
    return out


def _section(title: str, body: str) -> str:
    body = (body or "").strip()
    return f"\n\n{title}:\n{body}" if body else ""


def _fix_message(problems: list[str]) -> str:
    items = "\n".join(f"- {p}" for p in problems)
    return ("Your prompt failed these automated checks. Fix only these problems, keep everything else "
            "(dialogue verbatim, labels, style, subjects, story), and return the complete corrected prompt, nothing else:\n" + items)


def _write_loop(cfg, spec, user_msg, images, mode, duration, max_fix_rounds, strict, autofix, seed, silent_ok, pictures):
    messages = [{"role": "user", "content": user_msg}]
    log, best = [], None
    for rnd in range(max_fix_rounds + 1):
        raw = providers.chat(cfg, spec, messages, images, temperature=cfg.get("temperature", 0.4),
                             max_tokens=cfg.get("max_tokens", 4096), seed=seed + rnd)
        if not raw.strip():
            raise RuntimeError(f"{cfg.get('label')} returned an empty reply. For reasoning models raise max_tokens.")
        if rnd == 0 and looks_like_question(extract_prompt(raw)):
            raise RuntimeError(f"The model asked: {extract_prompt(raw)}\nAdd the answer to the inputs (idea, references, duration, dialogue...) and queue again.")
        r = lint(raw, mode, duration, autofix=autofix, silent_ok=silent_ok, pictures=pictures)
        problems = r.errors + (r.warnings if strict else [])
        log.append(f"round {rnd + 1}: {len(r.errors)} error(s), {len(r.warnings)} warning(s)")
        score = (len(r.errors), len(r.warnings))
        if best is None or score <= best[0]:
            best = (score, r)
        if not problems:
            break
        if rnd < max_fix_rounds:
            log[-1] += " -> asked the model to fix"
            messages += [{"role": "assistant", "content": r.text}, {"role": "user", "content": _fix_message(problems)}]
    r = best[1]
    return r, r.report() + "\n\nmodel: " + str(cfg.get("label")) + "\n" + "\n".join(log)


def _meta(r, cfg, **extra):
    return {"mode": r.mode, "duration": r.duration, "passed": r.passed, "errors": r.errors, "warnings": r.warnings,
            "model": cfg.get("label") if cfg else None, "created": _dt.datetime.now().isoformat(timespec="seconds"), **extra}


def _ui(prompt: str, report: str):
    return {"text": [prompt + "\n\n------------------------------\n" + report]}


def _images_b64(slots):
    return [providers.image_to_png_b64(im) for _, im in slots]


def _picture_set(mode, slots, references):
    if mode != REF_MODE:
        return None
    s = sorted({n for n, _ in slots} | set(_pictures_named(references)))
    return s or None


# ====================================================================== nodes

class MMH3ModelSelect:
    """Choose the writer model: presets from models.json plus every installed Ollama model."""

    @classmethod
    def _options(cls):
        conf = providers.load_models_config()
        labels = [p["label"] for p in conf.get("presets", []) if p.get("label")]
        if conf.get("discover_ollama", True):
            known = {p.get("model") for p in conf.get("presets", []) if p.get("provider") == "ollama"}
            for name in providers.discover_ollama(conf.get("ollama_base_url", providers.DEFAULT_BASE["ollama"])):
                if name not in known:
                    labels.append(f"Ollama - {name} (installed)")
        return labels or ["(edit mmh3_prompt/models.json)"]

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "preset": (cls._options(), {"tooltip": "Presets live in mmh3_prompt/models.json; installed Ollama models are listed automatically."}),
            "model_override": ("STRING", {"default": "", "tooltip": "Optional: any model name for this provider (e.g. an OpenRouter slug)."}),
            "base_url_override": ("STRING", {"default": "", "tooltip": "Optional: a different server URL (e.g. another machine's Ollama)."}),
            "temperature": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 2.0, "step": 0.05}),
            "max_tokens": ("INT", {"default": 4096, "min": 256, "max": 64000, "step": 256,
                                   "tooltip": "Raise for reasoning models that think before answering."}),
        }}

    RETURN_TYPES = ("MMH3_LLM",)
    RETURN_NAMES = ("llm",)
    FUNCTION = "select"
    CATEGORY = CATEGORY

    def select(self, preset, model_override, base_url_override, temperature, max_tokens):
        conf = providers.load_models_config()
        cfg = next((dict(p) for p in conf.get("presets", []) if p.get("label") == preset), None)
        m = re.match(r"^Ollama - (.+) \(installed\)$", preset)
        if cfg is None and m:
            name = m.group(1)
            cfg = {"label": preset, "provider": "ollama", "model": name}
            if any(h in name.lower() for h in conf.get("vision_hints", [])):
                cfg["vision"] = True
        if cfg is None:
            raise ValueError(f"Preset '{preset}' is not in mmh3_prompt/models.json on this machine.")
        if cfg.get("provider") == "ollama":
            for k, v in conf.get("ollama_defaults", {}).items():
                cfg.setdefault(k, v)
            cfg.setdefault("base_url", conf.get("ollama_base_url", providers.DEFAULT_BASE["ollama"]))
        if model_override.strip():
            cfg["model"] = model_override.strip()
            cfg["label"] = f"{cfg['label']} -> {cfg['model']}"
        if base_url_override.strip():
            cfg["base_url"] = base_url_override.strip()
        cfg["temperature"], cfg["max_tokens"] = temperature, max_tokens
        return (cfg,)


class MMH3PromptSpec:
    """The instructions the writer model follows (the MiniMax H3 guide), plus optional house rules."""

    @classmethod
    def INPUT_TYPES(cls):
        files = sorted(f for f in os.listdir(SPECS_DIR) if f.lower().endswith((".txt", ".md"))) if os.path.isdir(SPECS_DIR) else []
        return {"required": {
            "spec_file": ([AUTO_SPEC] + files, {"tooltip": "auto: the Ref2VA guide for Ref2VA, the base guide for T2VA/I2VA/FL2VA/L2VA. Add your own .txt to mmh3_prompt/specs/."}),
            "extra_rules": ("STRING", {"multiline": True, "default": "",
                                       "placeholder": "Optional house rules appended to the guide, e.g. 'Vertical 9:16 framing, phone-camera look.'"}),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("spec",)
    FUNCTION = "load"
    CATEGORY = CATEGORY

    def load(self, spec_file, extra_rules):
        rules = ""
        if extra_rules.strip():
            rules = ("\n\n\n==================================================\nHOUSE RULES (from the user)\n"
                     "==================================================\n\n" + extra_rules.strip())
        if spec_file == AUTO_SPEC:
            return (SPEC_SENTINEL + rules,)
        return (_read_spec(spec_file) + rules,)

    @classmethod
    def IS_CHANGED(cls, spec_file, extra_rules):
        try:
            names = SPEC_FOR_MODE.values() if spec_file == AUTO_SPEC else [spec_file]
            return tuple(os.path.getmtime(os.path.join(SPECS_DIR, n)) for n in set(names)), extra_rules
        except OSError:
            return float("nan")


_LOOP_INPUTS = {
    "max_fix_rounds": ("INT", {"default": 2, "min": 0, "max": 6, "tooltip": "How many times the model may fix check failures."}),
    "strict": ("BOOLEAN", {"default": False, "tooltip": "Also send warnings back for fixing, not just errors."}),
    "autofix": ("BOOLEAN", {"default": True, "tooltip": "Fix layout, section order and alignment lines without calling the model."}),
    "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF, "control_after_generate": True,
                     "tooltip": "Change for a different draft (same inputs + same seed reuse the cached result)."}),
}
_PICTURES = {f"picture_{i}": ("IMAGE",) for i in range(1, MAX_PICTURES + 1)}


class MMH3PromptWriter:
    """Write an H3 prompt with the chosen model, check it against the guide, and let the model fix what fails."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm": ("MMH3_LLM",),
                "spec": ("STRING", {"forceInput": True}),
                "mode": (list(MODES) + ["auto"], {"default": REF_MODE,
                         "tooltip": "Ref2VA: pictures are references (people, rooms, props, composition). auto: base modes by picture count (0 T2VA, 1 I2VA, 2 FL2VA)."}),
                "duration": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 120.0, "step": 0.01,
                                       "tooltip": "Total seconds. 0 = not set (needed for FL2VA / L2VA and multi-shot)."}),
                "idea": ("STRING", {"multiline": True, "default": "",
                                    "placeholder": "What happens in the clip: action, camera, mood, sound..."}),
                "references": ("STRING", {"multiline": True, "default": "",
                                          "placeholder": "Ref2VA: what each picture is, e.g.\nPicture 1: the woman - identity, hair, outfit\nPicture 2: the living room\nPicture 4: last frame of the previous clip - blocking only"}),
                **_LOOP_INPUTS,
            },
            "optional": {
                **_PICTURES,
                "dialogue": ("STRING", {"multiline": True, "default": "", "placeholder": "Exact spoken lines, verbatim (optional)"}),
                "visible_text": ("STRING", {"multiline": True, "default": "", "placeholder": "Exact on-screen text (optional)"}),
                "audio_references": ("STRING", {"multiline": True, "default": "",
                                                "placeholder": "Optional, e.g. 'Audio 1: voice timbre reference for the woman'"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "MMH3_META")
    RETURN_NAMES = ("prompt", "report", "meta")
    FUNCTION = "write"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True

    def write(self, llm, spec, mode, duration, idea, references, max_fix_rounds, strict, autofix, seed,
              picture_1=None, picture_2=None, picture_3=None, picture_4=None,
              dialogue="", visible_text="", audio_references=""):
        slots = _slots(picture_1, picture_2, picture_3, picture_4)
        mode = _resolve_mode(mode, len(slots))
        _check_duration(mode, duration)
        if mode == REF_MODE and not slots and not _pictures_named(references):
            raise ValueError("Ref2VA needs references: connect picture inputs, or name them in 'references' "
                             "(e.g. 'Picture 1: the woman, Picture 2: the living room').")
        if not idea.strip() and not slots:
            raise ValueError("Write something in 'idea' first.")
        vision = bool(llm.get("vision"))
        user = _context(mode, duration, slots, vision, references, audio_references)
        user += _section("REQUEST", idea or "Animate the references naturally.")
        user += _section("EXACT DIALOGUE (verbatim, each line exactly once, inside <d> blocks)", dialogue)
        user += _section("EXACT VISIBLE TEXT (verbatim, in double quotes)", visible_text)
        silent_ok = bool(re.search(r"\b(silent|no sound|silence)\b", idea, re.I))
        pictures = _picture_set(mode, slots, references)
        r, report = _write_loop(llm, _resolve_spec(spec, mode), user, _images_b64(slots), mode, float(duration),
                                max_fix_rounds, strict, autofix, seed, silent_ok, pictures)
        notes = []
        if slots and not vision:
            notes.append("note: this model cannot see images - it wrote from the 'references' text only.")
        if mode == "FL2VA" and len(slots) < 2:
            notes.append("note: FL2VA with fewer than 2 pictures connected.")
        if notes:
            report += "\n" + "\n".join(notes)
        meta = _meta(r, llm, idea=idea, references=references, dialogue=dialogue, visible_text=visible_text,
                     audio_references=audio_references, pictures=pictures, report=report)
        return {"ui": _ui(r.text, report), "result": (r.text, report, meta)}


SYMPTOMS = {
    "none (use feedback only)": "",
    "voice plays but the mouth stays still": "Move the spoken line into the first beat, add a clear mouth-movement phrase (for example 'her jaw and lips move clearly through every word') and remove any stillness or no-changes wording from the speaking shot.",
    "a line is spoken twice": "A sentence outside the <d> block (description, summary or sound sections) restates, previews or paraphrases the line. Delete it and describe only what is seen and heard.",
    "mouth keeps moving after the line": "Straight after </d>, describe the speaker closing their lips and returning to a non-speaking state.",
    "actions dropped / change every seed": "There are more distinct actions than seconds. Keep about one action per second; keep the visual detail but cut actions.",
    "click or noise burst at the start": "The soundscape is too empty. Give overall_soundscape a real soft ambience floor (room tone, air, a distant hum) and prefer a quiet pad over N/A in non_diegetic_music.",
    "wrong camera move": "The camera term does not match what should move. Re-check zoom vs push, pan vs truck and tilt vs pedestal; use one clear move and only the four allowed amplitude/speed phrases.",
    "too much motion": "Reduce motion: fewer, smaller actions, and a static shot or a camera move 'with small amplitude' / 'at slow speed'.",
    "too little motion": "Add clear visible actions with cause and effect, and a camera move, within about one action per second.",
    "identity drift (face / hair / outfit)": "Describe the referenced person's identity traits more concretely in subject_definitions and at their first appearance (face, hair, skin, outfit, accessories), and keep retention fully_preserved for them.",
    "room / environment drift": "Describe the environment subject more concretely (layout, key furniture and positions, materials, colours, light direction) and refer to it where the subject moves through it.",
    "copied the reference too literally (looks like a still)": "The references are being treated as frames. Make sure pictures that only define a person or place are cited inside a <Subject N> line, not given their own <Picture N> line, the summary is [reference generation], and the description describes new motion and composition.",
    "carry-over frame copied too literally / soft": "Scope the carry-over picture to composition only: its own line as a composition anchor that defines position, pose, body angle and camera framing, retention weak_reference, and state that appearance comes from the subject references.",
    "voice sounds wrong": "Describe the speaker's voice precisely at the first vocal event: age, gender, pitch, timbre, speaking rate, accent (or the <Audio N> timbre reference).",
    "sound effects doubled": "A sound appears both in a shot and in overall_soundscape. Keep moment-specific sounds in the shot only and continuous ambience in the soundscape only.",
}


class MMH3PromptRefine:
    """Revise a prompt from what the render actually did (pick a symptom and/or write feedback)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm": ("MMH3_LLM",),
                "spec": ("STRING", {"forceInput": True}),
                "prompt": ("STRING", {"forceInput": True}),
                "symptom": (list(SYMPTOMS.keys()),),
                "feedback": ("STRING", {"multiline": True, "default": "", "placeholder": "What should change? (optional if a symptom is picked)"}),
                **_LOOP_INPUTS,
            },
            "optional": {"meta": ("MMH3_META",), **_PICTURES},
        }

    RETURN_TYPES = ("STRING", "STRING", "MMH3_META")
    RETURN_NAMES = ("prompt", "report", "meta")
    FUNCTION = "refine"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True

    def refine(self, llm, spec, prompt, symptom, feedback, max_fix_rounds, strict, autofix, seed,
               meta=None, picture_1=None, picture_2=None, picture_3=None, picture_4=None):
        guidance = SYMPTOMS.get(symptom, "")
        if not guidance and not feedback.strip():
            raise ValueError("Pick a symptom or write feedback so the model knows what to change.")
        meta = dict(meta or {})
        mode, duration = infer_mode_and_duration(prompt)
        mode = meta.get("mode", mode)
        duration = float(meta.get("duration") or duration or 0.0)
        slots = _slots(picture_1, picture_2, picture_3, picture_4)
        refs = meta.get("references", "")
        pictures = _picture_set(mode, slots, refs) or meta.get("pictures")
        user = _context(mode, duration, slots, bool(llm.get("vision")), refs, meta.get("audio_references", ""))
        user += _section("CURRENT PROMPT (this was rendered)", prompt)
        if guidance:
            user += _section(f"PROBLEM SEEN IN THE RENDER: {symptom}", guidance)
        user += _section("USER FEEDBACK", feedback)
        user += "\n\nRevise the prompt to fix this. Keep everything unrelated unchanged (dialogue verbatim, labels, style, subjects). Return the complete revised prompt only."
        r, report = _write_loop(llm, _resolve_spec(spec, mode), user, _images_b64(slots), mode, duration,
                                max_fix_rounds, strict, autofix, seed, False, pictures)
        meta.update(_meta(r, llm, refined_from=symptom, feedback=feedback, pictures=pictures, report=report))
        return {"ui": _ui(r.text, report), "result": (r.text, report, meta)}


class MMH3PromptCheck:
    """Check (and tidy) any H3 prompt, including ones written by hand. No model needed."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "", "placeholder": "Paste or connect an H3 prompt"}),
                "mode": (["auto"] + list(MODES), {"default": "auto", "tooltip": "auto reads the mode from the prompt itself."}),
                "duration": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 120.0, "step": 0.01}),
                "reference_pictures": ("INT", {"default": 0, "min": 0, "max": 6,
                                               "tooltip": "Ref2VA: how many reference images the video node gets (0 = don't check picture numbers)."}),
                "autofix": ("BOOLEAN", {"default": True}),
            },
            "optional": {"meta": ("MMH3_META",)},
        }

    RETURN_TYPES = ("STRING", "STRING", "BOOLEAN")
    RETURN_NAMES = ("prompt", "report", "passed")
    FUNCTION = "check"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True

    def check(self, prompt, mode, duration, reference_pictures, autofix, meta=None):
        pictures = reference_pictures or None
        if meta:
            if mode == "auto":
                mode = meta.get("mode", "auto")
            duration = duration or float(meta.get("duration") or 0)
            pictures = pictures or meta.get("pictures")
        r = lint(prompt, mode, duration, autofix=autofix, pictures=pictures)
        rep = r.report()
        return {"ui": _ui(r.text, rep), "result": (r.text, rep, r.passed)}


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._ -]+", "_", name.strip()).strip(" ._") or "h3_prompt"


def _saved_names():
    if not os.path.isdir(SAVE_DIR):
        return []
    files = [f for f in os.listdir(SAVE_DIR) if f.endswith(".txt")]
    files.sort(key=lambda f: os.path.getmtime(os.path.join(SAVE_DIR, f)), reverse=True)
    return [f[:-4] for f in files]


def _read_saved(name: str) -> str:
    if name not in _saved_names():  # also keeps the name inside SAVE_DIR
        raise FileNotFoundError(f"No saved prompt '{name}' in {SAVE_DIR}. Save one with 'MMH3 Prompt Save', then press R.")
    with open(os.path.join(SAVE_DIR, name + ".txt"), "r", encoding="utf-8") as f:
        return f.read().strip()


def _read_meta(name: str) -> dict:
    try:
        with open(os.path.join(SAVE_DIR, name + ".json"), "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _mode_and_duration(prompt: str, meta: dict) -> tuple[str, float]:
    mode, dur = infer_mode_and_duration(prompt)
    return meta.get("mode", mode), float(meta.get("duration") or dur or 0.0)


def _overwrite_saved(name: str, prompt: str) -> str:
    """Replace an existing saved prompt's text and re-check it, so its .json report matches the new text."""
    _read_saved(name)
    prompt = prompt.strip()
    if not prompt:
        raise ValueError("Nothing to save: the text is empty.")
    m = _read_meta(name)
    r = lint(prompt, *_mode_and_duration(prompt, m), autofix=False)
    m.update(passed=r.passed, errors=r.errors, warnings=r.warnings, report=r.report(),
             saved=_dt.datetime.now().isoformat(timespec="seconds"))
    with open(os.path.join(SAVE_DIR, name + ".txt"), "w", encoding="utf-8") as f:
        f.write(prompt + "\n")
    with open(os.path.join(SAVE_DIR, name + ".json"), "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2, ensure_ascii=False)
    return r.report()


class MMH3PromptSave:
    """Save the prompt as <name>.txt (+ .json with mode, duration, model and check report) for later use."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"forceInput": True}),
                "name": ("STRING", {"default": "shot_01"}),
                "overwrite": ("BOOLEAN", {"default": False, "tooltip": "Off: adds _2, _3... instead of replacing."}),
            },
            "optional": {"meta": ("MMH3_META",), "report": ("STRING", {"forceInput": True})},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "save"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True

    def save(self, prompt, name, overwrite, meta=None, report=""):
        if not (prompt or "").strip():
            raise ValueError("Nothing to save: the prompt is empty.")
        os.makedirs(SAVE_DIR, exist_ok=True)
        base = final = _safe_name(name)
        if not overwrite:
            i = 2
            while os.path.exists(os.path.join(SAVE_DIR, final + ".txt")):
                final, i = f"{base}_{i}", i + 1
        m = dict(meta or {})
        if not m:
            mode, dur = infer_mode_and_duration(prompt)
            r = lint(prompt, mode, dur, autofix=False)
            m = {"mode": mode, "duration": dur, "passed": r.passed, "errors": r.errors, "warnings": r.warnings}
            report = report or r.report()
        m.update(name=final, saved=_dt.datetime.now().isoformat(timespec="seconds"))
        if report:
            m["report"] = report
        with open(os.path.join(SAVE_DIR, final + ".txt"), "w", encoding="utf-8") as f:
            f.write(prompt.strip() + "\n")
        with open(os.path.join(SAVE_DIR, final + ".json"), "w", encoding="utf-8") as f:
            json.dump(m, f, indent=2, ensure_ascii=False)
        msg = f"Saved as '{final}' in {SAVE_DIR}\n(passed checks: {m.get('passed')})"
        return {"ui": {"text": [msg + "\n\n" + prompt.strip()]}, "result": (prompt.strip(),)}


class MMH3PromptLoad:
    """Load a saved prompt into any workflow. Picking one copies it into 'text', where it can be edited before use."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"prompt_name": (_saved_names() or [NONE_SAVED],)},
                "optional": {"text": ("STRING", {"multiline": True, "default": "",
                                                 "tooltip": "Filled with the saved prompt when you pick one. Edit it freely; this text is what gets sent. Empty: the saved file is used as is."})}}

    RETURN_TYPES = ("STRING", "STRING", "FLOAT", "STRING")
    RETURN_NAMES = ("prompt", "mode", "duration", "report")
    FUNCTION = "load"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True  # runs even when nothing is wired yet

    @classmethod
    def IS_CHANGED(cls, prompt_name, text=""):
        p = os.path.join(SAVE_DIR, prompt_name + ".txt")
        return os.path.getmtime(p) if os.path.exists(p) else float("nan")

    def load(self, prompt_name, text=""):
        saved = _read_saved(prompt_name)
        meta = _read_meta(prompt_name)
        prompt = text.strip() or saved
        mode, dur = _mode_and_duration(prompt, meta)
        report = meta.get("report", "")
        if prompt != saved:
            report = "edited after loading\n" + lint(prompt, mode, dur, autofix=False).report()
        return prompt, mode, dur, report


if PromptServer is not None and getattr(PromptServer, "instance", None) is not None:
    @PromptServer.instance.routes.get("/mmh3/api/prompt")
    async def _mmh3_saved_prompt(request):
        """Lets MMH3 Prompt Load fill its text box as soon as a saved prompt is picked."""
        try:
            return web.json_response({"prompt": _read_saved(request.rel_url.query.get("name", ""))},
                                     headers={"Cache-Control": "no-store"})
        except FileNotFoundError as e:
            return web.json_response({"error": str(e)}, status=404)

    @PromptServer.instance.routes.post("/mmh3/api/prompt")
    async def _mmh3_overwrite_prompt(request):
        """MMH3 Prompt Load's 'save text over file' button."""
        data = await request.json()
        try:
            return web.json_response({"report": _overwrite_saved(data.get("name", ""), data.get("prompt", ""))})
        except FileNotFoundError as e:
            return web.json_response({"error": str(e)}, status=404)
        except ValueError as e:
            return web.json_response({"error": str(e)}, status=400)


NODE_CLASS_MAPPINGS = {
    "MMH3ModelSelect": MMH3ModelSelect,
    "MMH3PromptSpec": MMH3PromptSpec,
    "MMH3PromptWriter": MMH3PromptWriter,
    "MMH3PromptRefine": MMH3PromptRefine,
    "MMH3PromptCheck": MMH3PromptCheck,
    "MMH3PromptSave": MMH3PromptSave,
    "MMH3PromptLoad": MMH3PromptLoad,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MMH3ModelSelect": "MMH3 Model Select",
    "MMH3PromptSpec": "MMH3 Prompt Spec (guide)",
    "MMH3PromptWriter": "MMH3 Prompt Writer",
    "MMH3PromptRefine": "MMH3 Prompt Refine",
    "MMH3PromptCheck": "MMH3 Prompt Check",
    "MMH3PromptSave": "MMH3 Prompt Save",
    "MMH3PromptLoad": "MMH3 Prompt Load",
}
