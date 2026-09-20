"""Checks the graph mmh3_builder produces for the /mmh3 page.

Offline by default: every link points at a node that exists, segments chain
through each other's sampler latents, and the optional parts (LoRAs, chain
load/save, upscale) appear only when asked for.

With MMH3_COMFY_URL set (e.g. http://127.0.0.1:8189) it also checks every node
against the running server's /object_info -- class present, required inputs
given, no unknown inputs, combo values offered. Nothing is queued.
"""
import importlib.util
import json
import os
import random
import urllib.request

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("mmh3_builder", os.path.join(_ROOT, "mmh3_builder.py"))
B = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(B)

BASE = {
    "refs": [{"image": "IMG_6216B.png"}, {"image": "house.png"}, {"image": "IMG_7223.jpg"}],
    "blocks": [
        {"label": "subjects", "on": True, "where": "before", "text": "subject_definitions: A"},
        {"label": "off", "on": False, "where": "before", "text": "DISABLED"},
        {"label": "sound", "on": True, "where": "after", "text": "overall_soundscape: B"},
    ],
    "segments": [{"prompt": "shot one"}, {"prompt": "shot two", "random": False, "seed": 42},
                 {"prompt": "shot three"}],
}


def links(prompt):
    for nid, node in prompt.items():
        for name, v in node["inputs"].items():
            if isinstance(v, list) and len(v) == 2 and isinstance(v[1], int):
                yield nid, name, v


def test_frame_count():
    assert B.frame_count(8, 24) == 192        # the workflow's 8 s at 24 fps
    assert B.frame_count(5, 24) == 124
    assert B.frame_count(0, 24) == 5
    for s in (1, 2.5, 7, 10, 15):
        assert (B.frame_count(s, 24) - 5) % 17 == 0 and B.frame_count(s, 24) >= s * 24


def test_snap_dim():
    assert B.snap_dim(720, 32) == 736 and B.snap_dim(1280, 32) == 1280
    assert B.snap_dim(720, 16) == 720 and B.snap_dim(725, 16) == 720
    assert B.snap_dim(729, 16) == 736 and B.snap_dim(728, 16) == 736  # halves round up
    assert B.snap_dim(725, 0) == 725          # Off leaves it alone
    assert B.snap_dim(10, 64) == 64           # never below one step
    assert B.snap_dim(725, 7) == 736          # unknown step falls back to 32
    for step in (8, 16, 32, 64):
        for v in range(64, 2048, 37):
            assert B.snap_dim(v, step) % step == 0


def test_snap_applies_to_build():
    p, info = B.build(dict(BASE, width=1000, height=1777, snap=64), random.Random(1))
    assert (info["width"], info["height"]) == (1024, 1792)
    assert p["s1_r2v"]["inputs"]["width"] == 1024
    p, info = B.build(dict(BASE, width=1000, height=1777, snap=0), random.Random(1))
    assert (info["width"], info["height"]) == (1000, 1777)


def test_assemble():
    text = B.assemble_prompt(BASE["blocks"], "  the shot ")
    assert text == "subject_definitions: A\n\nthe shot\n\noverall_soundscape: B"


def test_graph_links_resolve():
    p, info = B.build(BASE, random.Random(1))
    for nid, name, (src, _) in links(p):
        assert src in p, "%s.%s -> missing %s" % (nid, name, src)
    assert info["length"] == 192
    assert info["seeds"][1] == 42 and len(info["seeds"]) == 3
    assert info["frames"] == 192 * 3 - 22 * 2


def test_segments_chain():
    p, _ = B.build(BASE, random.Random(1))
    assert "context_latent" not in p["s1_ctx"]["inputs"]      # fresh start
    assert p["s2_ctx"]["inputs"]["context_latent"] == ["s1_sample", 0]
    assert p["s3_ctx"]["inputs"]["context_latent"] == ["s2_sample", 0]
    assert p["ctx_save"]["inputs"]["latent"] == ["s3_sample", 0]
    assert p["ctx_save"]["inputs"]["clip_index"] == 1
    r2v = p["s2_r2v"]["inputs"]
    assert r2v["ref_images.ref_image_2"] == ["ref3", 0]
    assert "DISABLED" not in r2v["prompt"] and "shot two" in r2v["prompt"]
    assert p["join_v3"]["inputs"]["any1"] == ["join_v2", 0]
    assert p["save_up"]["inputs"]["audio"] == ["join_a3", 0]


def test_options():
    s = dict(BASE, load_index=3, save_index=4,
             loras=[{"name": "a.safetensors", "on": True, "strength": 0.5},
                    {"name": "b.safetensors", "on": False, "strength": 1},
                    {"name": "c.safetensors", "on": True, "strength": 0.8}],
             upscale={"enabled": False}, save_segments=False)
    p, info = B.build(s, random.Random(1))
    assert p["s1_ctx"]["inputs"]["context_latent"] == ["ctx_load", 0]
    assert p["ctx_load"]["inputs"]["clip_index"] == 3
    assert p["lora3"]["inputs"]["model"] == ["lora1", 0] and "lora2" not in p
    assert p["s1_guider"]["inputs"]["model"] == ["lora3", 0]
    assert not any(k.startswith("up_") or k == "save_up" for k in p)
    assert not any(k.endswith("_save") and k.startswith("s") for k in p)
    assert info["frames"] == 192 * 3 - 22 * 3

    one, _ = B.build(dict(BASE, segments=[{"prompt": "x"}], save_index=0))
    assert one["save_low"]["inputs"]["video_frames"] == ["s1_trim", 0]
    assert "ctx_save" not in one


def test_per_segment_latents():
    """A latent per segment, so a stopped run still has a continuation point."""
    p, info = B.build(dict(BASE, save_index=4, save_every_segment=True), random.Random(1))
    saves = {k: v["inputs"] for k, v in p.items() if v["class_type"].endswith("SaveLatent")}
    assert sorted(saves) == ["s1_ctxsave", "s2_ctxsave", "s3_ctxsave"]
    assert [saves[k]["clip_index"] for k in sorted(saves)] == [4, 5, 6]
    assert [saves[k]["latent"] for k in sorted(saves)] == [["s1_sample", 0], ["s2_sample", 0], ["s3_sample", 0]]
    assert info["save_index"] == 6          # what a following run continues from

    p, info = B.build(dict(BASE, save_index=4, save_every_segment=False))
    assert p["ctx_save"]["inputs"]["clip_index"] == 4 and info["save_index"] == 4

    p, info = B.build(dict(BASE, save_index=0, save_every_segment=True))
    assert not any(v["class_type"].endswith("SaveLatent") for v in p.values())
    assert info["save_index"] == 0


def test_errors():
    for bad in (dict(BASE, refs=[]), dict(BASE, segments=[]),
                dict(BASE, blocks=[], segments=[{"prompt": " "}]),
                dict(BASE, context_length="7")):
        try:
            B.build(bad)
        except B.BuildError:
            continue
        raise AssertionError("accepted %r" % bad)


def test_live_object_info():
    url = os.environ.get("MMH3_COMFY_URL")
    if not url:
        print("  (skipped: set MMH3_COMFY_URL to check against a running ComfyUI)")
        return
    info = json.load(urllib.request.urlopen(url.rstrip("/") + "/object_info", timeout=60))
    p, _ = B.build(dict(BASE, loras=B.DEFAULTS["loras"]), random.Random(1))
    problems = []
    for nid, node in p.items():
        spec = info.get(node["class_type"])
        if spec is None:
            problems.append("%s: unknown class %s" % (nid, node["class_type"]))
            continue
        req = spec["input"].get("required", {})
        opt = dict(spec["input"].get("optional", {}))
        opt.update(spec["input"].get("hidden", {}))
        given = node["inputs"]
        for name in req:
            if name not in given:
                problems.append("%s: missing required %s" % (nid, name))
        for name, value in given.items():
            base = name.split(".")[0]
            decl = req.get(name) or opt.get(name) or req.get(base) or opt.get(base)
            if decl is None:
                problems.append("%s: unknown input %s" % (nid, name))
                continue
            kind = decl[0]
            options = kind if isinstance(kind, list) else (
                decl[1].get("options") if kind == "COMBO" and len(decl) > 1 else None)
            if options and not isinstance(value, list) and value not in options \
                    and name != "image":
                problems.append("%s: %s=%r not offered" % (nid, name, value))
    assert not problems, "\n".join(problems)


def test_delete_paths(tmp=None):
    """The page's delete endpoints: which files they match, what they refuse."""
    import shutil
    import sys
    import tempfile
    import types

    root = tempfile.mkdtemp()
    out = os.path.join(root, "output")
    os.makedirs(os.path.join(out, "h3_context"))
    os.makedirs(os.path.join(root, "temp"))
    fp = types.ModuleType("folder_paths")
    fp.get_output_directory = lambda: out
    fp.get_temp_directory = lambda: os.path.join(root, "temp")
    fp.get_input_directory = lambda: root
    sys.modules["folder_paths"] = fp
    sys.modules.setdefault("server", types.ModuleType("server"))
    sys.modules["server"].PromptServer = None
    pkg = types.ModuleType("dihan_nodes_pkg")
    pkg.__path__ = [_ROOT]
    sys.modules["dihan_nodes_pkg"] = pkg
    sys.modules["dihan_nodes_pkg.mmh3_builder"] = B
    spec = importlib.util.spec_from_file_location(
        "dihan_nodes_pkg.mmh3_page", os.path.join(_ROOT, "mmh3_page.py"))
    page = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(page)

    try:
        names = ["clip_00001.safetensors", "clip_00002.safetensors",
                 "clip_00003_.safetensors", "keepme.safetensors", "notes.txt"]
        for n in names:
            open(os.path.join(out, "h3_context", n), "w").close()
        found = page._chain_latent_files("h3_context")
        assert sorted(os.path.basename(f) for f in found) == names[:3], found
        gone, failed = page._delete(found)
        assert len(gone) == 3 and not failed
        assert sorted(os.listdir(os.path.join(out, "h3_context"))) == ["keepme.safetensors", "notes.txt"]
        assert page._chain_latent_files("missing_folder") == []

        vid = os.path.join(out, "vids")
        os.makedirs(vid)
        open(os.path.join(vid, "a.mp4"), "w").close()
        assert page._output_file({"filename": "a.mp4", "subfolder": "vids", "type": "output"})
        assert page._output_file({"filename": "a.mp4", "subfolder": "vids", "type": "temp"}) is None
        for bad in ({"filename": "../../etc/passwd", "subfolder": "vids"},
                    {"filename": "a.mp4", "subfolder": "../.."},
                    {"filename": "/etc/passwd"}, {"filename": ""},
                    {"filename": "a.mp4", "subfolder": "vids", "type": "models"},
                    {"filename": "gone.mp4", "subfolder": "vids"}):
            assert page._output_file(bad) is None, bad
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
