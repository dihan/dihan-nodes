"""A focused web UI for the MiniMax H3 Ref2VA chain, served at ``/mmh3``.

Like the status page, it lives on ComfyUI's own server, so it follows whatever
port ComfyUI was launched with (``http://<ip>:8189/mmh3`` with ``--port 8189``).

The page only needs three things from here:

* ``GET  /mmh3/api/options``  model / LoRA / image lists and saved chain slots
* ``GET/POST /mmh3/api/settings``  the last-used settings, kept server side so
  a phone and a desktop see the same form
* ``POST /mmh3/api/build``  turn the form into an API graph (mmh3_builder)

The page then queues that graph through ComfyUI's own ``POST /prompt``, so it
goes through exactly the validation and hooks a canvas run does. Uploads,
progress over the websocket, history, /view and interrupt are stock API too.
"""

import json
import os
import re

from . import mmh3_builder

try:
    from aiohttp import web
    from server import PromptServer
except Exception:  # pragma: no cover - not running inside ComfyUI
    web = None
    PromptServer = None

HERE = os.path.dirname(os.path.abspath(__file__))
PAGE_PATH = os.path.join(HERE, "web", "mmh3.html")
STATE_DIR = os.path.join(HERE, ".state")
SETTINGS_PATH = os.path.join(STATE_DIR, "mmh3_settings.json")

BASE = "/mmh3"
IMAGE_EXT = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif")

# (node class, input name) pairs whose combo lists become the page's dropdowns.
_COMBOS = {
    "unet": ("UNETLoader", "unet_name"),
    "clip": ("CLIPLoader", "clip_name"),
    "vae": ("VAELoader", "vae_name"),
    "loras": ("LoraLoaderModelOnly", "lora_name"),
    "samplers": ("KSamplerSelect", "sampler_name"),
    "schedulers": ("BasicScheduler", "scheduler"),
    "seedvr2_dit": ("SeedVR2LoadDiTModel", "model"),
    "seedvr2_vae": ("SeedVR2LoadVAEModel", "model"),
}


def _combo(class_type, name):
    """Options of a combo input, read from the live node definition."""
    try:
        import nodes

        cls = nodes.NODE_CLASS_MAPPINGS[class_type]
        types = cls.INPUT_TYPES()
        spec = types.get("required", {}).get(name) or types.get("optional", {}).get(name)
        options = spec[0]
        if isinstance(options, str) and options == "COMBO":
            options = spec[1].get("options", [])
        return [str(o) for o in options]
    except Exception:
        return []


def _input_images():
    import folder_paths

    root = folder_paths.get_input_directory()
    out = []
    for dirpath, dirnames, files in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith(".") and d != "clipspace"]
        for f in files:
            if f.lower().endswith(IMAGE_EXT):
                full = os.path.join(dirpath, f)
                rel = os.path.relpath(full, root).replace("\\", "/")
                out.append((os.path.getmtime(full), rel))
    out.sort(reverse=True)
    return [rel for _, rel in out]


def _chain_slots(folder):
    """Indexed chain latents on disk: clip_00003.safetensors -> 3."""
    import folder_paths

    path = os.path.join(folder_paths.get_output_directory(), folder)
    slots = []
    try:
        for f in os.listdir(path):
            m = re.search(r"_(\d{5})\.safetensors$", f)
            if m:
                full = os.path.join(path, f)
                slots.append({"index": int(m.group(1)), "file": f,
                              "mtime": os.path.getmtime(full)})
    except OSError:
        pass
    return sorted(slots, key=lambda s: s["index"])


def _chain_latent_files(folder):
    """Every chain latent in `folder`, indexed slots and auto-numbered alike.

    Deliberately narrower than "everything in the folder": only files this
    pack's Save Latent node writes are matched, so a folder someone also keeps
    other work in survives a clear.
    """
    import folder_paths

    path = os.path.join(folder_paths.get_output_directory(), folder)
    try:
        names = os.listdir(path)
    except OSError:
        return []
    return [os.path.join(path, f) for f in sorted(names)
            if re.search(r"_\d{5}_?\.safetensors$", f)]


def _output_file(entry):
    """Resolve a /view-style {filename, subfolder, type} to a path on disk.

    Returns None for anything that escapes the output or temp folder -- the
    page sends back what history gave it, but this is a delete, so the path is
    re-derived and checked rather than trusted.
    """
    import folder_paths

    roots = {"output": folder_paths.get_output_directory(),
             "temp": folder_paths.get_temp_directory()}
    root = roots.get(entry.get("type") or "output")
    name = str(entry.get("filename") or "")
    if not root or not name or os.path.isabs(name):
        return None
    sub = str(entry.get("subfolder") or "")
    if os.path.isabs(sub):
        return None
    root = os.path.realpath(root)
    full = os.path.realpath(os.path.join(root, sub, name))
    if full != root and not full.startswith(root + os.sep):
        return None
    return full if os.path.isfile(full) else None


def _delete(paths):
    """Remove each path, reporting what went and what would not."""
    gone, failed = [], []
    for path in paths:
        try:
            os.remove(path)
            gone.append(os.path.basename(path))
        except OSError as exc:
            failed.append("%s: %s" % (os.path.basename(path), exc.strerror or exc))
    return gone, failed


def _read_settings():
    try:
        with open(SETTINGS_PATH, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _write_settings(data):
    os.makedirs(STATE_DIR, exist_ok=True)
    tmp = SETTINGS_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1)
    os.replace(tmp, SETTINGS_PATH)


def _register_routes():
    routes = PromptServer.instance.routes

    @routes.get(BASE)
    async def mmh3_page(request):
        try:
            with open(PAGE_PATH, "r", encoding="utf-8") as fh:
                body = fh.read()
        except OSError as exc:
            return web.Response(status=500, text="mmh3.html missing: %s" % exc)
        return web.Response(text=body, content_type="text/html",
                            headers={"Cache-Control": "no-store"})

    @routes.get(BASE + "/api/options")
    async def mmh3_options(request):
        folder = request.rel_url.query.get("folder") or mmh3_builder.DEFAULTS["latent_folder"]
        data = {name: _combo(*spec) for name, spec in _COMBOS.items()}
        data["images"] = _input_images()
        data["slots"] = _chain_slots(folder)
        data["defaults"] = mmh3_builder.DEFAULTS
        data["limits"] = {"segments": mmh3_builder.MAX_SEGMENTS, "refs": mmh3_builder.MAX_REFS}
        return web.json_response(data, headers={"Cache-Control": "no-store"})

    @routes.get(BASE + "/api/settings")
    async def mmh3_get_settings(request):
        return web.json_response({"settings": _read_settings()},
                                 headers={"Cache-Control": "no-store"})

    @routes.post(BASE + "/api/settings")
    async def mmh3_save_settings(request):
        try:
            data = await request.json()
            _write_settings(data)
        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=400)
        return web.json_response({"ok": True})

    @routes.post(BASE + "/api/latents/clear")
    async def mmh3_clear_latents(request):
        """Delete every saved chain latent, so the next run starts clean."""
        try:
            body = await request.json()
        except ValueError:
            body = {}
        folder = body.get("folder") or mmh3_builder.DEFAULTS["latent_folder"]
        if os.path.isabs(folder) or ".." in folder.split("/"):
            return web.json_response({"error": "Bad folder"}, status=400)
        gone, failed = _delete(_chain_latent_files(folder))
        return web.json_response({"deleted": gone, "failed": failed})

    @routes.post(BASE + "/api/files/delete")
    async def mmh3_delete_files(request):
        """Delete the mp4s of a run (joined, upscaled and per-segment clips)."""
        try:
            body = await request.json()
        except ValueError:
            return web.json_response({"error": "Body must be JSON"}, status=400)
        entries = body.get("files")
        if not isinstance(entries, list):
            return web.json_response({"error": "files must be a list"}, status=400)
        paths, failed = [], []
        for entry in entries:
            path = _output_file(entry) if isinstance(entry, dict) else None
            if path:
                paths.append(path)
            else:
                failed.append("%s: not in the output folder" % (entry or {}).get("filename"))
        gone, more = _delete(paths)
        return web.json_response({"deleted": gone, "failed": failed + more})

    @routes.post(BASE + "/api/build")
    async def mmh3_build(request):
        try:
            settings = await request.json()
        except ValueError:
            return web.json_response({"error": "Body must be JSON"}, status=400)
        try:
            prompt, info = mmh3_builder.build(settings)
        except (mmh3_builder.BuildError, ValueError, TypeError, KeyError) as exc:
            return web.json_response({"error": str(exc)}, status=400)
        # Deliberately no settings write here: the page posts a flattened
        # payload (one batch's segments, the chain slots it resolved), and
        # storing that would lose the batches it was flattened from. The page
        # saves the real settings through /api/settings as they change.
        return web.json_response({"prompt": prompt, "info": info})

    return BASE


def setup():
    if PromptServer is None or getattr(PromptServer, "instance", None) is None:
        return
    try:
        _register_routes()
        from .status_page import _server_url

        print("[dihan-nodes] MiniMax H3 page: %s" % _server_url(BASE))
    except Exception as exc:  # pragma: no cover
        print("[dihan-nodes] MiniMax H3 page disabled: %s" % exc)
