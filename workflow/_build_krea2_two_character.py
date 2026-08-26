"""Builds `Krea2 Two-Character Identity.json` in the ComfyUI 0.4 graph format.

The workflow is generated rather than hand-edited so that node sockets, widget
order and link bookkeeping stay in step with `krea2_two_character.py` — edit this,
re-run it, and `tests/test_krea2_workflow.py` will tell you if anything drifted.

    python3 workflow/_build_krea2_two_character.py "workflow/Krea2 Two-Character Identity.json"
    python3 tests/test_krea2_workflow.py
"""
import json, sys

CORE = {"cnr_id": "comfy-core", "ver": "0.26.0"}

nodes, links = [], []
_ids = {"node": 0, "link": 0}


def node(type_, pos, size, widgets=None, inputs=None, outputs=None, *,
         title=None, mode=0, props=None, color=None, bgcolor=None):
    _ids["node"] += 1
    n = {"id": _ids["node"], "type": type_, "pos": list(pos), "size": list(size),
         "flags": {}, "order": 0, "mode": mode,
         "inputs": inputs or [], "outputs": outputs or [],
         "properties": dict(props or {}, **{"Node name for S&R": type_}),
         "widgets_values": widgets if widgets is not None else []}
    if title:
        n["title"] = title
    if color:
        n["color"], n["bgcolor"] = color, bgcolor
    nodes.append(n)
    return n


def inp(name, type_, optional=False):
    d = {"name": name, "type": type_, "link": None}
    if optional:
        d["shape"] = 7
    return d


def widget_inp(name, type_):
    return {"name": name, "type": type_, "widget": {"name": name}, "link": None}


def out(name, type_):
    return {"name": name, "type": type_, "links": []}


def link(src, src_slot, dst, dst_name):
    """Connect src.outputs[src_slot] -> dst.inputs[named dst_name]."""
    _ids["link"] += 1
    lid = _ids["link"]
    o = src["outputs"][src_slot]
    i = next(x for x in dst["inputs"] if x["name"] == dst_name)
    assert i["link"] is None, f"{dst['type']}.{dst_name} already connected"
    assert o["type"] == i["type"], f"type mismatch {o['type']} -> {i['type']}"
    o["links"].append(lid)
    i["link"] = lid
    links.append([lid, src["id"], src_slot, dst["id"], dst["inputs"].index(i), o["type"]])


def note(title, pos, size, text, color="#432", bg="#653"):
    return node("Note", pos, size, [text], title=title, color=color, bgcolor=bg)


def markdown(title, pos, size, text):
    return note(title, pos, size, text, color="#323", bg="#535")


# ── 1 · MODELS ────────────────────────────────────────────────────────────────
unet = node("UNETLoader", (60, -150), (470, 130),
            ["krea2_turbo_fp8_scaled.safetensors", "default"],
            outputs=[out("MODEL", "MODEL")],
            props={**CORE, "models": [{"name": "krea2_turbo_fp8_scaled.safetensors",
                                       "url": "https://huggingface.co/Comfy-Org/Krea-2/resolve/main/diffusion_models/krea2_turbo_fp8_scaled.safetensors",
                                       "directory": "diffusion_models"}]})
lora = node("LoraLoaderModelOnly", (60, 20), (470, 120),
            ["Krea2/krea2_identity_edit_v1_2.safetensors", 1],
            inputs=[inp("model", "MODEL")], outputs=[out("MODEL", "MODEL")], props=CORE)
clip = node("CLIPLoader", (60, 190), (470, 140),
            ["qwen3vl_4b_fp8_scaled.safetensors", "krea2", "default"],
            outputs=[out("CLIP", "CLIP")],
            props={**CORE, "models": [{"name": "qwen3vl_4b_fp8_scaled.safetensors",
                                       "url": "https://huggingface.co/Comfy-Org/Krea-2/resolve/main/text_encoders/qwen3vl_4b_fp8_scaled.safetensors",
                                       "directory": "text_encoders"}]})
vae = node("VAELoader", (60, 380), (470, 90), ["qwen_image_vae.safetensors"],
           outputs=[out("VAE", "VAE")],
           props={**CORE, "models": [{"name": "qwen_image_vae.safetensors",
                                      "url": "https://huggingface.co/Comfy-Org/Krea-2/resolve/main/vae/qwen_image_vae.safetensors",
                                      "directory": "vae"}]})
note("Model files", (60, 520), (470, 210),
     "Four files, all auto-downloadable from the links on the loader nodes:\n\n"
     "- krea2_turbo_fp8_scaled  (diffusion_models)\n"
     "- qwen3vl_4b_fp8_scaled  (text_encoders) - type must be 'krea2'.\n"
     "  The vision tower matters: the encode node feeds both faces through it.\n"
     "- qwen_image_vae  (vae)\n"
     "- krea2_identity_edit_v1_2.safetensors  (loras)\n"
     "  from huggingface.co/conradlocke/krea2-identity-edit\n\n"
     "The LoRA is what these nodes patch. Without it you get a plain Krea 2 render.")

# ── 2 · CHARACTERS ────────────────────────────────────────────────────────────
img_a = node("LoadImage", (620, -150), (450, 400), ["character_a.png", "image"],
             outputs=[out("IMAGE", "IMAGE"), out("MASK", "MASK")], title="character A",
             props=CORE)
img_b = node("LoadImage", (620, 300), (450, 400), ["character_b.png", "image"],
             outputs=[out("IMAGE", "IMAGE"), out("MASK", "MASK")], title="character B",
             props=CORE)
note("Two people, one pass", (620, 740), (450, 250),
     "One clear photo per character. Head-and-shoulders or half-body reads best;\n"
     "a tiny face in a wide shot has few reference tokens to give.\n\n"
     "Both faces are placed in a SINGLE pass. If they still drift toward each\n"
     "other after tuning, the model card's own workaround is to chain: generate\n"
     "with A, then run that output back in as character_a and insert B.\n\n"
     "Slot A is REPRODUCED by the LoRA, slot B is RE-RENDERED into the scene.\n"
     "So A's likeness mostly comes for free and B's is the one you tune - that\n"
     "is what identity_b (the model card's ref_boost, ~4) is for.\n\n"
     "Wire the same images, in the same order, into the encode node.")

# ── 3 · INSTRUCTION ───────────────────────────────────────────────────────────
enc_p = node("Krea2TwoCharacterEncode", (1140, -150), (520, 330),
             ["Ada and Ben shaking hands in a sunlit office, Ada on the left, "
              "Ben on the right, both looking at the camera",
              "Ada", "Ben", 768, ""],
             inputs=[inp("clip", "CLIP"), inp("character_a", "IMAGE"), inp("character_b", "IMAGE")],
             outputs=[out("CONDITIONING", "CONDITIONING")], title="positive")
enc_n = node("Krea2TwoCharacterEncode", (1140, 230), (520, 250),
             ["", "Ada", "Ben", 768, ""],
             inputs=[inp("clip", "CLIP"), inp("character_a", "IMAGE"), inp("character_b", "IMAGE")],
             outputs=[out("CONDITIONING", "CONDITIONING")], title="negative (leave the prompt empty)")
note("Writing the instruction", (1140, 530), (520, 460),
     "Describe the SCENE you want, not the photos you fed in.\n\n"
     "name_a / name_b put 'This is Ada.' next to each image inside the prompt the\n"
     "vision encoder sees, so 'Ada on the left' binds to a specific face instead\n"
     "of the model guessing. Clear both names to get the exact template the LoRA\n"
     "was trained on.\n\n"
     "Say where each person is, and keep it consistent with the region masks in\n"
     "group 5 if you enable them.\n\n"
     "The negative node is the trained unconditional: EMPTY prompt, same two\n"
     "images, same names. It only matters at CFG > 1 - the turbo model runs at\n"
     "CFG 1, where the negative is ignored and you can bypass this node.\n\n"
     "grounding_px 768 is in-distribution (the LoRA trained on 384-768). Dropping\n"
     "to ~512 loosens the grip on the reference photos.")

# ── 4 · IDENTITY ──────────────────────────────────────────────────────────────
patch = node("Krea2TwoCharacterPatch", (1730, -150), (450, 400),
             [1.0, 4.0, "fit", 0.0, False, False],
             inputs=[inp("model", "MODEL"), inp("vae", "VAE"),
                     inp("character_a", "IMAGE"), inp("character_b", "IMAGE"),
                     inp("target_latent", "LATENT", True),
                     inp("face_mask_a", "MASK", True), inp("face_mask_b", "MASK", True),
                     inp("region_a", "MASK", True), inp("region_b", "MASK", True)],
             outputs=[out("MODEL", "MODEL")])
note("Tuning, in order", (1730, 600), (450, 560),
     "Change ONE thing at a time. The dials interact.\n\n"
     "1. Start where it is: identity_b 4.0 (the model card's strong-likeness\n"
     "   value), identity_a 1.0, no masks, 1-1.5 MP.\n\n"
     "   1.0 is the dial OFF, not a baseline. Creeping up from 1.0 in 0.1 steps\n"
     "   is how people conclude that B 'needs 10'.\n\n"
     "2. One face is weak -> move THAT dial in whole steps (1 -> 2 -> 4 -> 6),\n"
     "   leave the other alone. Past ~10 the card says the edit breaks down; if\n"
     "   B is still weak at 10, stop turning and go to step 3.\n\n"
     "3. The two faces blend into one person -> paint a face mask per character\n"
     "   (group 5) so the boost targets faces, not clothes and background.\n"
     "   Still blending -> isolate_references on.\n\n"
     "4. Faces swap sides, or both come out as the same person -> enable the\n"
     "   region masks (group 5) and set region_exclusivity 0.5-0.8.\n\n"
     "5. Result looks pasted together -> both dials BELOW 1.0.\n\n"
     "6. Stuck -> swap_reference_order on, then redo step 2. It puts the other\n"
     "   character in the copied slot, which tells you which face is easier to\n"
     "   preserve. Swap the images on the encode node to match.\n\n"
     "Also from the model card: grounding_px HIGHER = stronger identity (768,\n"
     "up to ~1024; lower it if you get doubled compositions), 12 steps favour\n"
     "face detail over 8, and 'fit' geometry needs the v1.2 weights - on v1.1\n"
     "use 'crop (legacy)' and match the source aspect ratio.\n\n"
     "Any dial off-neutral builds an L x L attention bias: slower, and a few\n"
     "hundred MB at ~1 MP. At 1.0 / 1.0 with nothing wired, it costs nothing.")

# ── 5 · MASKS (bypassed) ──────────────────────────────────────────────────────
region_img = node("LoadImage", (620, 1040), (450, 400), ["region_split.png", "image"],
                  outputs=[out("IMAGE", "IMAGE"), out("MASK", "MASK")],
                  title="region split (white = A's half)", mode=4, props=CORE)
to_mask = node("ImageToMask", (1140, 1040), (300, 90), ["red"],
               inputs=[inp("image", "IMAGE")], outputs=[out("MASK", "MASK")],
               title="region_a", mode=4, props=CORE)
inv_mask = node("InvertMask", (1140, 1180), (300, 60),
                inputs=[inp("mask", "MASK")], outputs=[out("MASK", "MASK")],
                title="region_b", mode=4, props=CORE)
note("5 - Masks (bypassed - Ctrl+B to enable)", (1490, 1040), (690, 470),
     "FACE MASKS - which part of a character's PHOTO the identity dial applies to.\n"
     "Right-click the character's LoadImage -> Open in MaskEditor, paint the face,\n"
     "then drag its MASK output to face_mask_a / face_mask_b. Feather the edge:\n"
     "soft masks ramp the boost instead of hard-gating it.\n"
     "An all-black mask silently turns that dial off - the node warns in the log.\n\n"
     "REGION MASKS - which part of the OUTPUT each character is allowed to drive.\n"
     "Load one black-and-white split image (white = A's side), ImageToMask gives\n"
     "region_a and InvertMask gives region_b. Rough halves are enough. Wire them\n"
     "to region_a / region_b, then raise region_exclusivity to 0.5-0.8 to actually\n"
     "suppress each character outside its own side. At 0.0 the regions only\n"
     "amplify and nothing is held apart.\n\n"
     "Keep the regions and the prompt agreeing with each other - masks that say\n"
     "left/right while the prompt says the opposite fight, and the prompt wins.\n\n"
     "Select these three nodes and press Ctrl+B to un-bypass them.")

# ── 6 · GENERATE ──────────────────────────────────────────────────────────────
latent = node("EmptySD3LatentImage", (2240, -150), (330, 120), [1024, 1024, 1],
              outputs=[out("LATENT", "LATENT")], props=CORE)
ks = node("KSampler", (2240, 20), (330, 300),
          [0, "randomize", 10, 1, "euler", "simple", 1],
          inputs=[inp("model", "MODEL"), inp("positive", "CONDITIONING"),
                  inp("negative", "CONDITIONING"), inp("latent_image", "LATENT")],
          outputs=[out("LATENT", "LATENT")], props=CORE)
dec = node("VAEDecode", (2240, 370), (330, 50),
           inputs=[inp("samples", "LATENT"), inp("vae", "VAE")],
           outputs=[out("IMAGE", "IMAGE")], props=CORE)
save = node("SaveImage", (2630, -150), (600, 640), ["krea2_two_character"],
            inputs=[inp("images", "IMAGE")], outputs=[out("images", "IMAGE")], props=CORE)
note("Sampling", (2240, 460), (330, 560),
     "Settings above are for the TURBO model: 10 steps, CFG 1, euler/simple.\n\n"
     "On raw krea2_raw_bf16 instead: ~28 steps, CFG ~4, and the negative encode\n"
     "node starts mattering.\n\n"
     "Prefer euler or another ODE sampler. er_sde injects noise that disrupts the\n"
     "reference-copy channel the whole method runs on.\n\n"
     "Generate at 2 MP or less. Above the LoRA's trained range the references\n"
     "start bleeding into the output and people duplicate.\n\n"
     "target_latent is wired from the same EmptySD3LatentImage that feeds the\n"
     "sampler. That is not decoration: it moves the VAE encode of both faces to\n"
     "node-execution time. Without it the encode happens on the first sampling\n"
     "step, where ComfyUI can evict part of the diffusion model to make room and\n"
     "leave the rest of the run streaming weights from CPU.\n\n"
     "Lock the seed once something works, then change one dial at a time.")

markdown("START HERE", (60, -560), (1600, 370),
         "KREA 2 - TWO-CHARACTER IDENTITY\n"
         "==============================\n\n"
         "Two people, both faces preserved, one pass. Load a photo of each character\n"
         "in group 2, write the scene in group 3, hit Run. Everything else is tuning.\n\n"
         "Needs the dihan-nodes pack (Krea2TwoCharacterPatch / Krea2TwoCharacterEncode)\n"
         "plus the four model files listed in group 1.\n\n"
         "The one thing worth knowing up front: the LoRA's two reference slots are not\n"
         "equal. Slot A is REPRODUCED near-pixel; slot B is RE-RENDERED into the new\n"
         "scene. So A's face tends to arrive on its own and B's is the one that needs\n"
         "the dial - identity_b ships at 4.0 here because that is the strong-likeness\n"
         "value in the LoRA's model card. identity_a starts at 1.0 (off).\n\n"
         "Full reference: docs/KREA2_TWO_CHARACTER.md")

# ── wiring ────────────────────────────────────────────────────────────────────
link(unet, 0, lora, "model")
link(lora, 0, patch, "model")
link(vae, 0, patch, "vae")
link(vae, 0, dec, "vae")
link(clip, 0, enc_p, "clip")
link(clip, 0, enc_n, "clip")

link(img_a, 0, enc_p, "character_a")
link(img_b, 0, enc_p, "character_b")
link(img_a, 0, enc_n, "character_a")
link(img_b, 0, enc_n, "character_b")
link(img_a, 0, patch, "character_a")
link(img_b, 0, patch, "character_b")

link(latent, 0, patch, "target_latent")
link(latent, 0, ks, "latent_image")
link(patch, 0, ks, "model")
link(enc_p, 0, ks, "positive")
link(enc_n, 0, ks, "negative")
link(ks, 0, dec, "samples")
link(dec, 0, save, "images")

link(region_img, 0, to_mask, "image")
link(to_mask, 0, inv_mask, "mask")
# region_a / region_b left unconnected on purpose: the group is bypassed, and an
# unpainted split image would hand the patch node two all-zero masks.

# ── execution order (topological; litegraph recomputes, but ship it sane) ──────
by_id = {n["id"]: n for n in nodes}
incoming = {n["id"]: {l[1] for l in links if l[3] == n["id"]} for n in nodes}
order, placed = 0, set()
while len(placed) < len(nodes):
    ready = [n for n in nodes if n["id"] not in placed and incoming[n["id"]] <= placed]
    assert ready, "cycle in the graph"
    for n in sorted(ready, key=lambda n: n["id"]):
        n["order"] = order
        order += 1
        placed.add(n["id"])

groups = [
    {"id": 1, "title": "1 - MODELS", "bounding": [40, -180, 510, 930], "color": "#3f789e", "flags": {}},
    {"id": 2, "title": "2 - CHARACTERS", "bounding": [600, -180, 490, 1190], "color": "#3f789e", "flags": {}},
    {"id": 3, "title": "3 - INSTRUCTION", "bounding": [1120, -180, 560, 1190], "color": "#3f789e", "flags": {}},
    {"id": 4, "title": "4 - IDENTITY", "bounding": [1710, -180, 490, 1350], "color": "#a1309b", "flags": {}},
    {"id": 5, "title": "5 - MASKS (bypassed)", "bounding": [600, 1010, 1600, 520], "color": "#88A", "flags": {}},
    {"id": 6, "title": "6 - GENERATE", "bounding": [2220, -180, 1030, 1250], "color": "#3f789e", "flags": {}},
]

for n in nodes:                      # litegraph convention: no links -> null, not []
    for o in n["outputs"]:
        if not o["links"]:
            o["links"] = None

graph = {
    "id": "3a7c9f10-2c4e-4d61-9b0a-krea2twochar",
    "revision": 0,
    "last_node_id": _ids["node"],
    "last_link_id": _ids["link"],
    "nodes": nodes,
    "links": links,
    "groups": groups,
    "config": {},
    "extra": {"ds": {"scale": 0.55, "offset": [120, 640]}},
    "version": 0.4,
}
path = sys.argv[1]
with open(path, "w") as f:
    json.dump(graph, f, indent=2)
print(f"wrote {path}: {len(nodes)} nodes, {len(links)} links")
