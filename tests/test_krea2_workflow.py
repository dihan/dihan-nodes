"""Validates workflow/Krea2 Two-Character Identity.json against the real node
definitions and against litegraph's link bookkeeping.

A workflow that references a socket the node doesn't have, or lists widget values
in the wrong order, fails silently-ish in the browser (dropped links, widgets
reading each other's values), so it is worth checking here rather than in ComfyUI.
"""
import importlib.util
import json
import os
import sys
import types

import torch  # noqa: F401  (the node module imports it)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_NODE = os.path.join(_ROOT, "krea2_two_character.py")
_WF = os.path.join(_ROOT, "workflow", "Krea2 Two-Character Identity.json")

for name in ("comfy", "comfy.patcher_extension", "comfy.utils", "comfy.ldm",
             "comfy.ldm.common_dit", "comfy.ldm.flux", "comfy.ldm.flux.layers"):
    sys.modules.setdefault(name, types.ModuleType(name))
sys.modules["comfy.ldm.flux.layers"].timestep_embedding = lambda *a, **k: None
try:
    import einops  # noqa: F401
except ImportError:
    e = types.ModuleType("einops"); e.rearrange = lambda *a, **k: None
    sys.modules["einops"] = e

_spec = importlib.util.spec_from_file_location("k2", _NODE)
k2 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(k2)

wf = json.load(open(_WF))
nodes = {n["id"]: n for n in wf["nodes"]}
assert len(nodes) == len(wf["nodes"]), "duplicate node ids"

# ── link bookkeeping ─────────────────────────────────────────────────────────
seen = set()
for lid, src_id, src_slot, dst_id, dst_slot, ltype in wf["links"]:
    assert lid not in seen, f"duplicate link id {lid}"
    seen.add(lid)
    src, dst = nodes[src_id], nodes[dst_id]
    o, i = src["outputs"][src_slot], dst["inputs"][dst_slot]
    assert lid in (o["links"] or []), f"link {lid} missing from {src['type']}.{o['name']}"
    assert i["link"] == lid, f"{dst['type']}.{i['name']} does not point back at link {lid}"
    assert o["type"] == i["type"] == ltype, f"link {lid} type mismatch"
for n in wf["nodes"]:
    for o in n["outputs"]:
        for lid in (o["links"] or []):
            assert lid in seen, f"{n['type']}.{o['name']} references unknown link {lid}"
    for i in n["inputs"]:
        assert i["link"] is None or i["link"] in seen, f"{n['type']}.{i['name']} dangling link"
assert wf["last_link_id"] >= max(seen), "last_link_id below the highest link"
assert wf["last_node_id"] >= max(nodes), "last_node_id below the highest node id"
print(f"1 ok  {len(wf['links'])} links consistent in both directions")

# ── execution order is topological ───────────────────────────────────────────
for lid, src_id, _s, dst_id, _d, _t in wf["links"]:
    assert nodes[src_id]["order"] < nodes[dst_id]["order"], \
        f"{nodes[src_id]['type']} runs after {nodes[dst_id]['type']}"
print("2 ok  execution order is topological")

# ── our nodes match their INPUT_TYPES ────────────────────────────────────────
LINKED = {"MODEL", "VAE", "CLIP", "IMAGE", "MASK", "LATENT", "CONDITIONING"}

def schema(cls):
    d = cls.INPUT_TYPES()
    req, opt = d.get("required", {}), d.get("optional", {})
    names = list(req) + list(opt)
    sockets = [n for n in names if isinstance((req | opt)[n][0], str) and (req | opt)[n][0] in LINKED]
    widgets = [n for n in names if n not in sockets]
    return req, opt, sockets, widgets, (req | opt)

checked = 0
for n in wf["nodes"]:
    cls = k2.NODE_CLASS_MAPPINGS.get(n["type"])
    if cls is None:
        continue
    checked += 1
    req, opt, sockets, widgets, spec = schema(cls)

    for i in n["inputs"]:
        assert i["name"] in spec, f"{n['type']}: input '{i['name']}' is not in INPUT_TYPES"
        assert i["name"] in sockets, f"{n['type']}: '{i['name']}' is a widget, not a socket"
        assert i["type"] == spec[i["name"]][0], f"{n['type']}.{i['name']} type mismatch"
        assert (i.get("shape") == 7) == (i["name"] in opt), \
            f"{n['type']}.{i['name']}: optional sockets need shape 7, required ones must not have it"
    wired = {i["name"] for i in n["inputs"]}
    for name in sockets:
        if name in req:
            assert name in wired, f"{n['type']}: required socket '{name}' missing from the graph"
            assert next(i for i in n["inputs"] if i["name"] == name)["link"] is not None, \
                f"{n['type']}: required socket '{name}' is not connected"

    assert [o["type"] for o in n["outputs"]] == list(cls.RETURN_TYPES), \
        f"{n['type']}: outputs do not match RETURN_TYPES"

    vals = n["widgets_values"]
    assert len(vals) == len(widgets), \
        f"{n['type']}: {len(vals)} widget values for {len(widgets)} widgets {widgets}"
    for name, val in zip(widgets, vals):
        cfg = spec[name]
        opts = cfg[1] if len(cfg) > 1 else {}
        if isinstance(cfg[0], list):
            assert val in cfg[0], f"{n['type']}.{name}: {val!r} not in {cfg[0]}"
        elif cfg[0] == "FLOAT":
            assert isinstance(val, (int, float)) and not isinstance(val, bool), f"{n['type']}.{name}"
            assert opts.get("min", -1e9) <= val <= opts.get("max", 1e9), f"{n['type']}.{name} out of range"
        elif cfg[0] == "INT":
            assert isinstance(val, int) and not isinstance(val, bool), f"{n['type']}.{name}"
            assert opts.get("min", -1 << 62) <= val <= opts.get("max", 1 << 62), f"{n['type']}.{name} out of range"
        elif cfg[0] == "BOOLEAN":
            assert isinstance(val, bool), f"{n['type']}.{name}: {val!r} is not a bool"
        elif cfg[0] == "STRING":
            assert isinstance(val, str), f"{n['type']}.{name}: {val!r} is not a string"
    assert n["properties"]["Node name for S&R"] == n["type"]
assert checked == 3, f"expected 3 of our nodes in the workflow, found {checked}"
print(f"3 ok  {checked} pack nodes match INPUT_TYPES (sockets, order, widget values)")

# ── the wiring the docs promise ──────────────────────────────────────────────
def find(type_, title=None):
    hits = [n for n in wf["nodes"] if n["type"] == type_ and (title is None or n.get("title") == title)]
    assert len(hits) == 1, f"expected exactly one {type_} {title or ''}, found {len(hits)}"
    return hits[0]

def src_of(n, socket):
    i = next(x for x in n["inputs"] if x["name"] == socket)
    l = next(l for l in wf["links"] if l[0] == i["link"])
    return nodes[l[1]], l[2]

patch = find("Krea2TwoCharacterPatch")
enc_p = find("Krea2TwoCharacterEncode", "positive")
enc_n = find("Krea2TwoCharacterEncode", "negative (leave the prompt empty)")
ks, latent = find("KSampler"), find("EmptySD3LatentImage")

# same images, same order, on the patch and both encode nodes
for slot in ("character_a", "character_b"):
    a = src_of(patch, slot)
    assert src_of(enc_p, slot) == a and src_of(enc_n, slot) == a, \
        f"{slot} differs between the patch and encode nodes"
assert src_of(patch, "character_a") != src_of(patch, "character_b"), "both slots share one image"
print("4 ok  both characters reach the patch and both encode nodes in the same order")

# target_latent is the SAME latent the sampler gets, which is the whole point
assert src_of(patch, "target_latent")[0]["id"] == latent["id"] == src_of(ks, "latent_image")[0]["id"], \
    "target_latent must come from the same EmptySD3LatentImage as KSampler.latent_image"
assert src_of(ks, "model")[0]["id"] == patch["id"], "KSampler must sample the patched model"
assert src_of(ks, "positive")[0]["id"] == enc_p["id"]
assert src_of(ks, "negative")[0]["id"] == enc_n["id"]
print("5 ok  target_latent shares the sampler's latent; sampler reads the patched model")

# the trained unconditional: empty prompt, same images, same names
assert enc_n["widgets_values"][0] == "", "the negative encode must have an empty prompt"
assert enc_p["widgets_values"][1:] == enc_n["widgets_values"][1:], \
    "negative encode must match the positive except for the prompt"
assert enc_p["widgets_values"][0].strip(), "the positive prompt is empty"
print("6 ok  negative encode is the trained unconditional")

# ships neutral, so a first run reproduces the upstream baseline
iv = patch["widgets_values"]
assert iv[0] == 1.0 and iv[1] == 1.0, "identity dials should ship neutral"
assert iv[2] == "fit" and iv[3] == 0.0 and iv[4] is False and iv[5] is False
print("7 ok  patch node ships neutral (1.0 / 1.0, fit, no exclusivity, no toggles)")

# the mask group is bypassed and hands nothing to the patch node
for slot in ("face_mask_a", "face_mask_b", "region_a", "region_b"):
    i = next(x for x in patch["inputs"] if x["name"] == slot)
    assert i["link"] is None, f"{slot} is wired; an unpainted mask would silently disable a dial"
bypassed = [n for n in wf["nodes"] if n["mode"] == 4]
assert {n["type"] for n in bypassed} == {"LoadImage", "ImageToMask", "InvertMask"}, \
    [n["type"] for n in bypassed]
print(f"8 ok  mask group is bypassed ({len(bypassed)} nodes) and no mask reaches the patch node")

# sampler settings match what the notes claim
seed, ctrl, steps, cfg, sampler, sched, denoise = ks["widgets_values"]
assert sampler == "euler", "er_sde and friends disrupt the reference-copy channel"
assert (steps, cfg, denoise) == (10, 1, 1), (steps, cfg, denoise)
w, h, batch = latent["widgets_values"]
assert w * h <= 2048 * 1024, f"{w}x{h} is above the LoRA's trained range"
print(f"9 ok  sampling {steps} steps / cfg {cfg} / {sampler} at {w}x{h}")

# every group actually contains nodes, and every node sits inside a group
def inside(n, g):
    gx, gy, gw, gh = g["bounding"]
    x, y = n["pos"]
    return gx <= x <= gx + gw and gy <= y <= gy + gh
for g in wf["groups"]:
    assert any(inside(n, g) for n in wf["nodes"]), f"group {g['title']!r} is empty"
loose = [n["type"] for n in wf["nodes"] if not any(inside(n, g) for g in wf["groups"])]
assert loose == ["Note"], f"nodes outside every group: {loose}"   # the START HERE banner
print(f"10 ok {len(wf['groups'])} groups laid out, only the banner sits outside")
print("\nall checks passed")
