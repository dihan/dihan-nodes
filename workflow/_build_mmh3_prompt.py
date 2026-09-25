"""Generates the MMH3 prompt-writer workflows (ComfyUI UI format).

    python3 workflow/_build_mmh3_prompt.py            # writes both files next to this script
    python3 workflow/_build_mmh3_prompt.py <out_dir>

Re-run after changing the inputs of the MMH3 prompt nodes; tests/test_mmh3_prompt.py checks
the result against INPUT_TYPES.
"""
import json
import os
import sys

BUILDER = "MMH3 Prompt Builder.json"
LOADER = "MMH3 Prompt Loader.json"

IDEA = ("She walks from the window to the camera, stops close to the lens and speaks one line, then smiles. "
        "Handheld phone-video feel, soft afternoon light.")
REFERENCES = ("Picture 1: the woman - identity, face, hair, body, outfit\n"
              "Picture 2: the living room - layout, furniture, materials, colours, light")
GREEN, PURPLE = ("#232", "#353"), ("#323", "#535")


class Graph:
    def __init__(self):
        self.nodes, self.links, self.groups, self.lid = [], [], [], 0

    def node(self, nid, type_, pos, size, widgets, inputs=(), outputs=(), title=None, mode=0, color=None):
        n = {"id": nid, "type": type_, "pos": list(pos), "size": list(size), "flags": {}, "order": len(self.nodes),
             "mode": mode, "inputs": [{"name": a, "type": t, "link": None} for a, t in inputs],
             "outputs": [{"name": a, "type": t, "links": [], "slot_index": i} for i, (a, t) in enumerate(outputs)],
             "properties": {"Node name for S&R": type_}, "widgets_values": widgets}
        if title:
            n["title"] = title
        if color:
            n["color"], n["bgcolor"] = color
        self.nodes.append(n)

    def link(self, a, a_slot, b, b_input):
        self.lid += 1
        na = next(n for n in self.nodes if n["id"] == a)
        nb = next(n for n in self.nodes if n["id"] == b)
        idx = next(i for i, x in enumerate(nb["inputs"]) if x["name"] == b_input)
        na["outputs"][a_slot]["links"].append(self.lid)
        nb["inputs"][idx]["link"] = self.lid
        self.links.append([self.lid, a, a_slot, b, idx, na["outputs"][a_slot]["type"]])

    def group(self, title, x, y, w, h, color="#3f789e"):
        self.groups.append({"title": title, "bounding": [x, y, w, h], "color": color, "font_size": 22, "flags": {}})

    def dump(self, path):
        wf = {"last_node_id": max(n["id"] for n in self.nodes), "last_link_id": self.lid, "nodes": self.nodes,
              "links": self.links, "groups": self.groups, "config": {},
              "extra": {"ds": {"scale": 0.75, "offset": [40, 40]}}, "version": 0.4}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(wf, f, indent=1, ensure_ascii=False)


LLM = [("llm", "MMH3_LLM")]
WRITER_OUT = [("prompt", "STRING"), ("report", "STRING"), ("meta", "MMH3_META")]
PICS = [(f"picture_{i}", "IMAGE") for i in range(1, 5)]
IMG_OUT = [("IMAGE", "IMAGE"), ("MASK", "MASK")]
SAVE_IN = [("prompt", "STRING"), ("meta", "MMH3_META"), ("report", "STRING")]


def builder(path):
    g = Graph()
    g.group("1 · Writer model", 20, 20, 420, 250)
    g.node(1, "MMH3ModelSelect", (40, 70), (380, 180), ["Claude Sonnet 5 (Anthropic API)", "", "", 0.4, 4096], outputs=LLM)
    g.group("2 · Guide (auto = Ref2VA guide in Ref2VA mode)", 20, 290, 420, 250)
    g.node(2, "MMH3PromptSpec", (40, 340), (380, 180), ["(auto: match mode)", ""], outputs=[("spec", "STRING")])

    g.group("3 · References = Reference to Video slots (Ctrl+B to use 3 and 4)", 20, 560, 840, 700)
    g.node(3, "LoadImage", (40, 610), (390, 320), ["example.png", "image"], outputs=IMG_OUT, title="Picture 1 (person)")
    g.node(4, "LoadImage", (450, 610), (390, 320), ["example.png", "image"], outputs=IMG_OUT, title="Picture 2 (room)")
    g.node(10, "LoadImage", (40, 950), (390, 290), ["example.png", "image"], outputs=IMG_OUT, title="Picture 3 (optional)", mode=4)
    g.node(11, "LoadImage", (450, 950), (390, 290), ["example.png", "image"], outputs=IMG_OUT,
           title="Picture 4 (carry-over frame, optional)", mode=4)

    g.group("4 · Describe the shot → write + check", 460, 20, 520, 520)
    g.node(5, "MMH3PromptWriter", (880, 70), (500, 1180),
           ["Ref2VA", 8.0, IDEA, REFERENCES, 2, False, True, 0, "fixed", "", "", ""],
           inputs=LLM + [("spec", "STRING")] + PICS, outputs=WRITER_OUT, color=GREEN)
    g.groups[-1]["bounding"] = [860, 20, 540, 1240]

    g.group("5 · Save for later (ComfyUI/user/h3_prompts)", 1420, 20, 460, 520)
    g.node(6, "MMH3PromptSave", (1440, 70), (420, 450), ["shot_01", False], inputs=SAVE_IN, outputs=[("prompt", "STRING")])

    g.group("6 · Optional: refine after a render (select both, Ctrl+M to enable)", 1420, 560, 960, 700, color="#8A8")
    g.node(7, "MMH3PromptRefine", (1440, 610), (440, 630),
           ["voice plays but the mouth stays still", "", 2, False, True, 0, "fixed"],
           inputs=LLM + [("spec", "STRING"), ("prompt", "STRING"), ("meta", "MMH3_META")] + PICS,
           outputs=WRITER_OUT, mode=2, color=PURPLE)
    g.node(8, "MMH3PromptSave", (1900, 610), (440, 440), ["shot_01_v2", False], inputs=SAVE_IN,
           outputs=[("prompt", "STRING")], title="MMH3 Prompt Save (refined)", mode=2)

    g.node(9, "Note", (1900, 70), (480, 470), [
        "MMH3 PROMPT BUILDER (Ref2VA)\n\n"
        "1. Pick the writer model. Installed Ollama models appear automatically; API presets need a key "
        "(env var or mmh3_prompt/api_keys.json).\n"
        "2. Guide: auto uses the Ref2VA guide for Ref2VA and the base guide for T2VA/I2VA/FL2VA/L2VA.\n"
        "3. Pictures are numbered by slot, same as the Reference to Video node's ref_image slots. "
        "Say what each one is in 'references'. Images are optional: naming them in 'references' is enough "
        "to write the prompt before the pictures exist.\n"
        "4. Write the idea, exact dialogue and on-screen text. Queue.\n"
        "   The writer drafts, the checker tests it against the guide's hard rules, and failures go back to the "
        "model (max_fix_rounds). The report on the node shows what passed.\n"
        "5. Saved to ComfyUI/user/h3_prompts/<name>.txt. Load it anywhere with 'MMH3 Prompt Load' (press R).\n\n"
        "New draft: change the seed. After a render: enable group 6, pick what went wrong, queue."], outputs=[])

    for a, b in ((1, "llm"), (2, "spec")):
        g.link(a, 0, 5, b)
        g.link(a, 0, 7, b)
    for src, pic in ((3, "picture_1"), (4, "picture_2"), (10, "picture_3"), (11, "picture_4")):
        g.link(src, 0, 5, pic)
        g.link(src, 0, 7, pic)
    g.link(5, 0, 6, "prompt"); g.link(5, 2, 6, "meta"); g.link(5, 1, 6, "report")
    g.link(5, 0, 7, "prompt"); g.link(5, 2, 7, "meta")
    g.link(7, 0, 8, "prompt"); g.link(7, 2, 8, "meta"); g.link(7, 1, 8, "report")
    g.dump(path)


def loader(path):
    g = Graph()
    g.group("Saved H3 prompt → Reference to Video", 20, 20, 900, 460)
    g.node(1, "MMH3PromptLoad", (40, 70), (460, 390), ["(no saved prompts yet)", ""],
           outputs=[("prompt", "STRING"), ("mode", "STRING"), ("duration", "FLOAT"), ("report", "STRING")], color=GREEN)
    g.node(2, "Note", (520, 70), (380, 390), [
        "Copy this node into your H3 workflow.\n\n"
        "Connect 'prompt' to the prompt input of MiniMax H3 Reference to Video (drag onto the text box). "
        "Feed the reference images in the same slot order the prompt was written for.\n\n"
        "'duration' is the saved clip length in seconds.\n\n"
        "Picking a saved prompt copies it into the text box; edit it there before queueing. Picking it again reloads the file.\n\n"
        "Press R after saving new prompts."],
           outputs=[])
    g.dump(path)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
    builder(os.path.join(out, BUILDER))
    loader(os.path.join(out, LOADER))
    print("wrote", os.path.join(out, BUILDER), "and", os.path.join(out, LOADER))
