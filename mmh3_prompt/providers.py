"""Tiny, dependency-free chat clients for the writer model.

Three wire formats cover almost everything:
  ollama         -> local Ollama (/api/chat)
  openai_compat  -> OpenAI, OpenRouter, LM Studio, llama.cpp server, vLLM, Groq,
                    Gemini (OpenAI-compatible endpoint), DeepSeek, Mistral, ...
  anthropic      -> Claude (Messages API)

A "config" is a plain dict built by the MMH3 Model Select node:
  {label, provider, model, base_url, api_key_env, vision, keep_alive, options, ...}
"""

from __future__ import annotations

import base64
import io
import json
import os
import urllib.error
import urllib.request

PACK_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_JSON = os.path.join(PACK_DIR, "models.json")
KEYS_JSON = os.path.join(PACK_DIR, "api_keys.json")

DEFAULT_BASE = {
    "ollama": "http://127.0.0.1:11434",
    "openai_compat": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com",
}


# ---------------------------------------------------------------- config files

def load_models_config() -> dict:
    try:
        with open(MODELS_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:  # keep the node usable even with a broken file
        print(f"[dihan-nodes] mmh3_prompt: could not read models.json: {e}")
        return {"presets": []}


def discover_ollama(base_url: str, timeout: float = 1.5) -> list[str]:
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/api/tags", timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
        return sorted(m.get("name") for m in data.get("models", []) if m.get("name"))
    except Exception:
        return []


def resolve_api_key(cfg: dict) -> str:
    if cfg.get("provider") == "ollama":
        return ""
    env = cfg.get("api_key_env") or ""
    if env and os.environ.get(env):
        return os.environ[env].strip()
    try:
        with open(KEYS_JSON, "r", encoding="utf-8") as f:
            keys = json.load(f)
        for k in (cfg.get("label"), env, cfg.get("key_name"), cfg.get("provider")):
            if k and keys.get(k):
                return str(keys[k]).strip()
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"[dihan-nodes] mmh3_prompt: could not read api_keys.json: {e}")
    if cfg.get("key_optional"):
        return ""
    raise RuntimeError(
        f"No API key for '{cfg.get('label')}'. Set the environment variable {env or '(api_key_env in models.json)'} "
        f"before starting ComfyUI, or add it to {KEYS_JSON} as {{\"{env or cfg.get('provider')}\": \"...\"}}."
    )


# ---------------------------------------------------------------- images

def image_to_png_b64(img, max_side: int = 1024) -> str:
    """Accepts a ComfyUI IMAGE tensor [B,H,W,C] (first frame used), a numpy array or a PIL image."""
    import numpy as np
    from PIL import Image

    if isinstance(img, Image.Image):
        pil = img
    else:
        arr = img
        if hasattr(arr, "detach"):
            arr = arr.detach().cpu().numpy()
        arr = np.asarray(arr)
        if arr.ndim == 4:
            arr = arr[0]
        if arr.dtype != np.uint8:
            arr = (np.clip(arr, 0.0, 1.0) * 255.0).round().astype(np.uint8)
        if arr.shape[-1] == 1:
            arr = arr[..., 0]
        pil = Image.fromarray(arr)
    pil = pil.convert("RGB")
    w, h = pil.size
    s = max(w, h)
    if s > max_side:
        pil = pil.resize((round(w * max_side / s), round(h * max_side / s)), Image.LANCZOS)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------- http

def _post(url: str, body: dict, headers: dict, timeout: float) -> dict:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **headers}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:800]
        raise RuntimeError(f"{url} returned HTTP {e.code}: {detail}") from None
    except urllib.error.URLError as e:
        raise RuntimeError(f"Could not reach {url}: {e.reason}. Is the server running and the base_url right?") from None


# ---------------------------------------------------------------- chat

def chat(cfg: dict, system: str, messages: list[dict], images_b64: list[str] | None = None,
         temperature: float = 0.4, max_tokens: int = 4096, seed: int | None = None, timeout: float = 300.0) -> str:
    """messages: [{role: user|assistant, content: str}]. Images attach to the FIRST user message."""
    provider = cfg.get("provider", "openai_compat")
    base = (cfg.get("base_url") or DEFAULT_BASE.get(provider, "")).rstrip("/")
    model = cfg.get("model")
    if not model:
        raise RuntimeError("No model name set. Pick a preset or fill model_override.")
    images_b64 = images_b64 or []
    if images_b64 and not cfg.get("vision", False):
        images_b64 = []  # caller already warned; never send pictures to a text-only model

    if provider == "ollama":
        msgs = [{"role": "system", "content": system}]
        for i, m in enumerate(messages):
            mm = {"role": m["role"], "content": m["content"]}
            if i == 0 and images_b64:
                mm["images"] = images_b64
            msgs.append(mm)
        opts = {"temperature": temperature, "num_predict": max_tokens, "num_ctx": int(cfg.get("num_ctx", 16384))}
        if seed is not None:
            opts["seed"] = int(seed) % (2**31)
        opts.update(cfg.get("options", {}))
        body = {"model": model, "messages": msgs, "stream": False, "options": opts,
                "keep_alive": cfg.get("keep_alive", 0)}
        if "think" in cfg:
            body["think"] = cfg["think"]
        out = _post(base + "/api/chat", body, {}, timeout)
        return (out.get("message") or {}).get("content", "")

    if provider == "anthropic":
        key = resolve_api_key(cfg)
        msgs = []
        for i, m in enumerate(messages):
            if i == 0 and images_b64:
                parts = [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b}} for b in images_b64]
                parts.append({"type": "text", "text": m["content"]})
                msgs.append({"role": m["role"], "content": parts})
            else:
                msgs.append({"role": m["role"], "content": m["content"]})
        body = {"model": model, "system": system, "messages": msgs, "max_tokens": max_tokens}
        if not cfg.get("no_temperature"):
            body["temperature"] = temperature
        body.update(cfg.get("options", {}))
        out = _post(base + "/v1/messages", body,
                    {"x-api-key": key, "anthropic-version": cfg.get("anthropic_version", "2023-06-01")}, timeout)
        return "".join(p.get("text", "") for p in out.get("content", []) if p.get("type") == "text")

    # openai-compatible
    key = resolve_api_key(cfg)
    msgs = [{"role": "system", "content": system}]
    for i, m in enumerate(messages):
        if i == 0 and images_b64:
            parts = [{"type": "text", "text": m["content"]}]
            parts += [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + b}} for b in images_b64]
            msgs.append({"role": m["role"], "content": parts})
        else:
            msgs.append({"role": m["role"], "content": m["content"]})
    body = {"model": model, "messages": msgs, cfg.get("max_tokens_param", "max_tokens"): max_tokens}
    if not cfg.get("no_temperature"):
        body["temperature"] = temperature
    if seed is not None and cfg.get("send_seed", True):
        body["seed"] = int(seed) % (2**31)
    body.update(cfg.get("options", {}))
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    headers.update(cfg.get("headers", {}))
    out = _post(base + "/chat/completions", body, headers, timeout)
    try:
        content = out["choices"][0]["message"]["content"]
    except Exception:
        raise RuntimeError(f"Unexpected response from {base}: {json.dumps(out)[:600]}") from None
    if isinstance(content, list):
        content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return content or ""
