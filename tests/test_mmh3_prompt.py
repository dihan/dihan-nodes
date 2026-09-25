"""Offline checks for the MMH3 prompt writer (mmh3_prompt/).

No ComfyUI, GPU or API key needed: the checker runs on fixed prompts, and the three
LLM wire formats (Ollama, OpenAI-compatible, Anthropic) are exercised against a small
local mock server that returns a flawed Ref2VA draft first and a clean one once it
receives the fix-up message. The shipped workflows are checked against INPUT_TYPES.

    python3 tests/test_mmh3_prompt.py
"""
import json
import os
import re
import shutil
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
_SAVE = tempfile.mkdtemp(prefix="h3_prompts_")
os.environ["H3_PROMPTS_DIR"] = _SAVE

from mmh3_prompt import lint as L          # noqa: E402
from mmh3_prompt import nodes as N         # noqa: E402
from mmh3_prompt import providers as P     # noqa: E402

GOOD = """subject_definitions:
<Subject 1> is the woman in <Picture 1>, in her early thirties, with shoulder-length wavy auburn hair, light freckles, a cream ribbed knit jumper, and small gold hoop earrings.
<Subject 2> is the living-room environment in <Picture 2>, with a pale grey linen sofa, a round mustard-yellow ottoman, a walnut dining table, oak floorboards, and daylight from a tall window on the left.

summary:
[reference generation] <Subject 1> stands by the ottoman in <Subject 2>, speaks one line to camera and smiles.

retention_analysis:
<Subject 1> (appears in [Shot 1]): fully_preserved - her face, freckles, auburn hair, cream jumper and gold hoops are retained.
<Subject 2> (appears in [Shot 1]): fully_preserved - the sofa, ottoman, dining table, oak floor and window light are retained.

detailed_description:
The target video is live-action and cinematic, with soft natural daylight and a warm, slightly desaturated palette.
[Shot 1] A medium close-up frames <Subject 1>, the woman with wavy auburn hair and light freckles in a cream ribbed knit jumper, standing in front of the mustard-yellow ottoman in <Subject 2>, the grey linen sofa soft behind her and window light from the left catching the side of her face. She looks into the lens, her lips part, and <Subject 1> (S1), speaking in a warm, low female voice with a relaxed London accent, says, <d>[English] Come in, it's warmer by the window.</d> Her jaw and lips move clearly through every word. She closes her lips into an easy smile and turns her head toward the window, her hair swinging across her shoulder. The camera pushes in with small amplitude at slow speed, keeping the walnut dining table softly out of focus on the right while the oak floor and warm window light stay consistent behind her.

overall_soundscape:
Quiet indoor room tone with faint traffic beyond the window and light fabric movement continues throughout.

non_diegetic_music:
A soft sustained synth pad at a slow tempo holds quietly underneath and fades out on the final frame."""

BAD = "Sure! Here you go:\n```\n" + GOOD.replace(
    "[reference generation] <Subject 1> stands", "[reference generation] <Subject 1> tells us to come in by the window as she stands").replace(
    "The camera pushes in with small amplitude at slow speed", "The camera slowly pushes in").replace(
    "Quiet indoor room tone", "Clean audio with no background noise, quiet indoor room tone") + "\n```\nLet me know!"


# ------------------------------------------------------------------ checker

def test_good_ref2va_passes():
    r = L.lint(GOOD, "Ref2VA", 6.0, pictures=2)
    assert r.passed and not r.warnings, r.report()
    assert r.text == GOOD, "a clean prompt must come back unchanged"


def test_bad_ref2va_errors():
    r = L.lint(BAD, "Ref2VA", 6.0, pictures=2)
    errs = "\n".join(r.errors)
    for needle in ("summary", "slowly", "asks for silence"):
        assert needle in errs, (needle, r.report())
    assert any("outside the prompt" in f for f in r.fixes)


def test_ref2va_structure_rules():
    broken = GOOD.replace("[reference generation]", "[reference gen]").replace(
        "fully_preserved - her face", "kept - her face").replace("<Picture 2>", "<Picture 5>")
    r = L.lint(broken, "Ref2VA", 6.0, pictures=[1, 2, 4])
    errs = "\n".join(r.errors)
    assert "Unknown summary task type" in errs and "marker 'kept'" in errs, r.report()
    assert "<Picture 5> is used but no such reference image" in errs, r.report()
    assert any("<Picture 2> is connected but never cited" in w for w in r.warnings)


def test_ref2va_autofix_layout_and_alignment():
    # summary after retention, a base-mode alignment line on top, header values on the same line
    s = GOOD.split("\n\n")
    reordered = "\n\n".join([s[0], s[2], s[1]] + s[3:]).replace("overall_soundscape:\n", "overall_soundscape: ")
    r = L.lint("How the reference pictures align with the target video \u2014 x.\n\n" + reordered, "Ref2VA", 6.0, pictures=2)
    assert r.passed, r.report()
    assert r.text == GOOD
    assert any("alignment line" in f for f in r.fixes) and any("reordered" in f for f in r.fixes)


def test_ref2va_rejects_base_field():
    r = L.lint(GOOD.replace("detailed_description:", "integrated_multimodal_description:"), "Ref2VA", 6.0)
    assert any("detailed_description" in e for e in r.errors), r.report()


def test_head_tilt_is_not_a_camera_move():
    r = L.lint(GOOD.replace("turns her head toward the window", "tilts her head slightly toward the window"), "Ref2VA", 6.0, pictures=2)
    assert r.passed, r.report()


def test_base_mode_fl2va():
    base = ("```\nintegrated_multimodal_description: [Shot 1] Live-action, cinematic, a woman turns toward the window. "
            "The camera pushes in with small amplitude at slow speed. The scene settles into the pose, spacing, viewpoint, "
            "lighting and composition established by Picture 2.\n\noverall_soundscape: Quiet room tone.\n\n"
            "non_diegetic_music: N/A\n```")
    r = L.lint(base, "FL2VA", 8.0)
    assert r.passed, r.report()
    assert r.text.startswith(L.alignment_line("FL2VA", 8.0, 1)), r.text


def test_mode_inference():
    assert L.infer_mode_and_duration(GOOD) == ("Ref2VA", 0.0)
    assert L.infer_mode_and_duration(L.alignment_line("L2VA", 6.0, 1) + "\n\nintegrated_multimodal_description: x")[0] == "L2VA"


def test_question_is_reported():
    r = L.lint("What is the total video duration in seconds?", "Ref2VA", 0)
    assert r.errors and "asked a question" in r.errors[0]


# ------------------------------------------------------------------ providers + writer loop

LOG = []


def _reply(msgs):
    last = msgs[-1]["content"]
    if isinstance(last, list):
        last = " ".join(p.get("text", "") for p in last if isinstance(p, dict))
    return "```\n" + GOOD + "\n```" if "failed these automated checks" in last else BAD


class _Mock(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        self._send({"models": [{"name": "qwen2.5vl:7b"}, {"name": "llava:13b"}]} if self.path == "/api/tags" else {}, 200)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        LOG.append({"path": self.path, "headers": dict(self.headers), "body": body})
        if self.path == "/api/chat":
            return self._send({"message": {"content": _reply(body["messages"])}})
        if self.path.endswith("/chat/completions"):
            if self.headers.get("Authorization") != "Bearer test-key":
                return self._send({"error": "bad key"}, 401)
            return self._send({"choices": [{"message": {"content": _reply(body["messages"])}}]})
        if self.path == "/v1/messages":
            if self.headers.get("x-api-key") != "test-key":
                return self._send({"error": "bad key"}, 401)
            return self._send({"content": [{"type": "text", "text": _reply(body["messages"])}]})
        self._send({}, 404)


def _server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Mock)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, "http://127.0.0.1:%d" % srv.server_address[1]


def test_writer_all_providers_and_fix_loop():
    import numpy as np

    srv, url = _server()
    os.environ["MMH3_TEST_KEY"] = "test-key"
    conf_path = os.path.join(_SAVE, "models.json")
    conf = json.load(open(P.MODELS_JSON))
    conf["ollama_base_url"] = url
    json.dump(conf, open(conf_path, "w"))
    orig = P.MODELS_JSON
    P.MODELS_JSON = conf_path
    try:
        labels = N.MMH3ModelSelect.INPUT_TYPES()["required"]["preset"][0]
        assert "Ollama - llava:13b (installed)" in labels, labels   # discovered; qwen2.5vl is already a preset
        spec = N.MMH3PromptSpec().load("(auto: match mode)", "Vertical 9:16.")[0]
        img = np.random.rand(1, 64, 48, 3).astype("float32")
        sel = N.MMH3ModelSelect().select
        cfgs = {
            "ollama": sel("Ollama - llava:13b (installed)", "", "", 0.4, 4096)[0],
            "openai_compat": dict(sel("OpenRouter (any model - set model_override)", "some/model", url + "/v1", 0.4, 4096)[0],
                                  api_key_env="MMH3_TEST_KEY"),
            "anthropic": dict(sel("Claude Sonnet 5 (Anthropic API)", "", url, 0.4, 4096)[0], api_key_env="MMH3_TEST_KEY"),
        }
        for name, cfg in cfgs.items():
            LOG.clear()
            out = N.MMH3PromptWriter().write(cfg, spec, "Ref2VA", 6.0, "She speaks to camera.",
                                             "Picture 1: the woman\nPicture 2: the living room", 2, False, True, 7,
                                             picture_1=img, picture_2=img, dialogue="Come in, it's warmer by the window.")
            prompt, report, meta = out["result"]
            assert len(LOG) == 2, (name, len(LOG), report)           # draft + one fix round
            assert meta["passed"] and prompt == GOOD, (name, report)
            b = LOG[0]["body"]
            sys_text = b.get("system") or b["messages"][0]["content"]
            assert "Ref2VA Prompt Writer" in sys_text and "HOUSE RULES" in sys_text, name   # auto spec resolved
            user = b["messages"][0] if name == "anthropic" else b["messages"][1]
            n_img = (len(user.get("images", [])) if name == "ollama" else
                     sum(1 for p in user["content"] if p["type"] in ("image", "image_url")))
            assert n_img == 2, (name, n_img)
            text = user["content"] if isinstance(user["content"], str) else " ".join(
                p.get("text", "") for p in user["content"] if isinstance(p, dict))
            assert "<Picture 1>, <Picture 2>" in text and "Picture 1: the woman" in text, text[:400]
            print("   %-14s 2 calls, %d images, passed" % (name, n_img))
        assert LOG[0]["body"]["model"] == "claude-sonnet-5"

        # no images at all: naming the pictures in 'references' is enough to write ahead of time
        LOG.clear()
        out = N.MMH3PromptWriter().write(cfgs["anthropic"], spec, "Ref2VA", 6.0, "x", "Picture 1: woman, Picture 2: room",
                                         1, False, True, 1)
        assert out["result"][2]["pictures"] == [1, 2] and out["result"][2]["passed"]

        try:
            N.MMH3PromptWriter().write(cfgs["anthropic"], spec, "Ref2VA", 6.0, "x", "", 1, False, True, 1)
            raise AssertionError("Ref2VA without references should fail")
        except ValueError as e:
            assert "references" in str(e)
        try:
            N.MMH3PromptWriter().write(cfgs["anthropic"], spec, "L2VA", 0.0, "x", "", 1, False, True, 1, picture_1=img)
            raise AssertionError("L2VA without duration should fail")
        except ValueError as e:
            assert "duration" in str(e)

        # refine keeps mode/pictures from meta
        LOG.clear()
        meta = out["result"][2]
        ref = N.MMH3PromptRefine().refine(cfgs["anthropic"], spec, GOOD, "a line is spoken twice", "", 1, False, True, 3, meta=meta)
        assert ref["result"][2]["passed"] and "PROBLEM SEEN IN THE RENDER" in LOG[0]["body"]["messages"][0]["content"]
    finally:
        P.MODELS_JSON = orig
        srv.shutdown()


def test_save_load_roundtrip():
    save = N.MMH3PromptSave()
    save.save(GOOD, "living room", False, {"mode": "Ref2VA", "duration": 6.0, "passed": True})
    save.save(GOOD, "living room", False, None)
    names = N.MMH3PromptLoad.INPUT_TYPES()["required"]["prompt_name"][0]
    assert set(names) >= {"living room", "living room_2"}, names
    prompt, mode, dur, _ = N.MMH3PromptLoad().load("living room")
    assert prompt == GOOD and mode == "Ref2VA" and dur == 6.0
    meta2 = json.load(open(os.path.join(_SAVE, "living room_2.json")))
    assert meta2["mode"] == "Ref2VA" and meta2["passed"], meta2


# ------------------------------------------------------------------ workflows vs INPUT_TYPES

def _widgets(cls):
    it = cls.INPUT_TYPES()
    out = []
    for sect in ("required", "optional"):
        for name, spec in it.get(sect, {}).items():
            kind = spec[0]
            is_widget = isinstance(kind, list) or (kind in ("STRING", "INT", "FLOAT", "BOOLEAN")
                                                   and not (len(spec) > 1 and spec[1].get("forceInput")))
            if is_widget:
                out.append((name, spec))
                if len(spec) > 1 and spec[1].get("control_after_generate"):
                    out.append(("control_after_generate", None))
    return out


def _sockets(cls):
    it = cls.INPUT_TYPES()
    out = []
    for sect in ("required", "optional"):
        for name, spec in it.get(sect, {}).items():
            kind = spec[0]
            if not isinstance(kind, list) and (kind not in ("STRING", "INT", "FLOAT", "BOOLEAN")
                                               or (len(spec) > 1 and spec[1].get("forceInput"))):
                out.append((name, kind))
    return out


def test_workflows_match_nodes():
    import importlib.util

    spec = importlib.util.spec_from_file_location("b", os.path.join(_ROOT, "workflow", "_build_mmh3_prompt.py"))
    b = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(b)
    tmp = tempfile.mkdtemp()
    try:
        for fname, make in ((b.BUILDER, b.builder), (b.LOADER, b.loader)):
            make(os.path.join(tmp, fname))
            shipped = os.path.join(_ROOT, "workflow", fname)
            assert json.load(open(os.path.join(tmp, fname))) == json.load(open(shipped)), \
                f"{fname} is stale; re-run workflow/_build_mmh3_prompt.py"
            wf = json.load(open(shipped))
            by_id = {n["id"]: n for n in wf["nodes"]}
            for n in wf["nodes"]:
                cls = N.NODE_CLASS_MAPPINGS.get(n["type"])
                if not cls:
                    continue
                ws = _widgets(cls)
                assert len(n["widgets_values"]) == len(ws), (fname, n["type"], n["widgets_values"], [w[0] for w in ws])
                for (name, spec_), val in zip(ws, n["widgets_values"]):
                    if spec_ and isinstance(spec_[0], list) and name != "prompt_name" and name != "preset":
                        assert val in spec_[0], (n["type"], name, val)
                    if spec_ and spec_[0] in ("INT", "FLOAT") and len(spec_) > 1:
                        assert spec_[1].get("min", val) <= val <= spec_[1].get("max", val), (n["type"], name, val)
                assert [(i["name"], i["type"]) for i in n["inputs"]] == _sockets(cls), (fname, n["type"])
                assert [(o["name"], o["type"]) for o in n["outputs"]] == list(zip(cls.RETURN_NAMES, cls.RETURN_TYPES))
            presets = [p["label"] for p in P.load_models_config()["presets"]]
            for n in wf["nodes"]:
                if n["type"] == "MMH3ModelSelect":
                    assert n["widgets_values"][0] in presets
            for lid, a, a_slot, t, t_slot, typ in wf["links"]:
                assert lid in by_id[a]["outputs"][a_slot]["links"], lid
                assert by_id[t]["inputs"][t_slot]["link"] == lid, lid
                assert by_id[t]["inputs"][t_slot]["type"] == typ == by_id[a]["outputs"][a_slot]["type"], lid
            print("   %s: %d nodes, %d links consistent" % (fname, len(wf["nodes"]), len(wf["links"])))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_guides_ship():
    names = os.listdir(N.SPECS_DIR)
    assert set(N.SPEC_FOR_MODE.values()) <= set(names), names
    guide = open(os.path.join(N.SPECS_DIR, "minimax_h3_ref2va_guide.txt")).read()
    ex = re.search(r"COMPLETE EXAMPLE.*?\n\n(subject_definitions:.*?)\n\n\n=====", guide, re.S).group(1)
    r = L.lint(ex, "Ref2VA", 6.0, pictures=2)
    assert r.passed and not r.warnings, "the guide's own example must pass the checker:\n" + r.report()


if __name__ == "__main__":
    try:
        for name, fn in sorted(globals().items()):
            if name.startswith("test_") and callable(fn):
                fn()
                print("ok", name)
    finally:
        shutil.rmtree(_SAVE, ignore_errors=True)
