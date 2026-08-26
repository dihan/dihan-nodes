import importlib.util
import math
import os
import sys
import types

import torch

_NODE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "krea2_two_character.py")

# stub the comfy surface the module imports at load time
for name in ("comfy", "comfy.patcher_extension", "comfy.utils", "comfy.ldm",
             "comfy.ldm.common_dit", "comfy.ldm.flux", "comfy.ldm.flux.layers"):
    sys.modules.setdefault(name, types.ModuleType(name))
sys.modules["comfy.ldm.flux.layers"].timestep_embedding = lambda *a, **k: None
try:
    import einops  # noqa
except ImportError:
    e = types.ModuleType("einops"); e.rearrange = lambda *a, **k: None
    sys.modules["einops"] = e

spec = importlib.util.spec_from_file_location(
    "k2", _NODE)
k2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(k2)

TXT, LA, LB, TGT = 3, 4, 4, 6          # tiny sequence: [text | refA | refB | target]
L = TXT + LA + LB + TGT
OA, OB, R0 = TXT, TXT + LA, TXT + LA + LB
dev, dt = torch.device("cpu"), torch.float32

def spec_(boost=1.0, face=None, region=None, excl=0.0):
    return {"boost": boost, "face_w": face, "region_w": region, "excl": excl}

def build(specs, isolate=False):
    return k2._identity_bias(specs, TXT, [LA, LB], TGT, L, isolate, dev, dt)

# 1. fully neutral -> no bias tensor at all (fast path preserved)
assert build([spec_(), spec_()]) is None, "neutral should skip the bias"
print("1 ok  neutral -> None")

# 2. plain boosts land only on target rows, in the right column blocks
b = build([spec_(boost=2.0), spec_(boost=0.5)])
assert torch.allclose(b[0, 0, R0:, OA:OA+LA], torch.full((TGT, LA), math.log(2.0)))
assert torch.allclose(b[0, 0, R0:, OB:OB+LB], torch.full((TGT, LB), math.log(0.5)))
assert b[0, 0, :R0].abs().max() == 0, "non-target rows must stay unbiased"
assert b[0, 0, R0:, :TXT].abs().max() == 0, "text columns must stay unbiased"
assert b[0, 0, R0:, R0:].abs().max() == 0, "target->target must stay unbiased"
print("2 ok  boosts -> correct blocks only")

# 3. soft face mask ramps the boost; a half-covered token gets half of it
face = torch.tensor([1.0, 0.5, 0.0, 1.0])
b = build([spec_(boost=3.0, face=face), spec_()])
row = b[0, 0, R0, OA:OA+LA]
assert torch.allclose(row, torch.log(torch.tensor([3.0, 2.0, 1.0, 3.0])), atol=1e-6)
assert b[0, 0, R0:, OB:OB+LB].abs().max() == 0
print("3 ok  soft face mask -> 1+(b-1)*w")

# 4. region + exclusivity: boosted inside the region, cut off outside it
region = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
b = build([spec_(boost=2.0, region=region, excl=1.0), spec_()])
col = b[0, 0, R0:, OA]
assert torch.allclose(col[:3], torch.full((3,), math.log(2.0)), atol=1e-6)
assert torch.allclose(col[3:], torch.full((3,), math.log(1e-4)), atol=1e-6)
# excl=0 -> region only amplifies, never suppresses
b0 = build([spec_(boost=2.0, region=region, excl=0.0), spec_()])
c0 = b0[0, 0, R0:, OA]
assert torch.allclose(c0[:3], torch.full((3,), math.log(2.0)), atol=1e-6)
assert c0[3:].abs().max() == 0
# excl alone (boost 1.0) still gates, and still builds a tensor
b1 = build([spec_(boost=1.0, region=region, excl=0.5), spec_()])
c1 = b1[0, 0, R0:, OA]
assert c1[:3].abs().max() == 0 and torch.allclose(c1[3:], torch.full((3,), math.log(0.5)), atol=1e-6)
print("4 ok  region exclusivity gates outside the region")

# 5. isolation blinds the refs to each other, nothing else
b = build([spec_(), spec_()], isolate=True)
assert b is not None
assert (b[0, 0, OA:OA+LA, OB:OB+LB] == k2.NEG).all()
assert (b[0, 0, OB:OB+LB, OA:OA+LA] == k2.NEG).all()
assert b[0, 0, OA:OA+LA, OA:OA+LA].abs().max() == 0, "self-attention within a ref stays open"
assert b[0, 0, OA:OA+LA, :TXT].abs().max() == 0, "refs keep reading the text"
assert b[0, 0, OA:OA+LA, R0:].abs().max() == 0, "refs keep reading the target"
assert b[0, 0, R0:, OA:OB+LB].abs().max() == 0, "isolation alone must not bias the target"
print("5 ok  isolation is ref<->ref only")

# 6. no fully-masked row anywhere -> softmax can never NaN
worst = build([spec_(boost=0.0, region=torch.zeros(TGT), excl=1.0),
               spec_(boost=0.0, region=torch.zeros(TGT), excl=1.0)], isolate=True)
assert torch.isfinite(worst).all()
assert (worst.max(dim=-1).values > k2.NEG / 2).all(), "every row keeps an open column"
print("6 ok  no all-masked rows under the most extreme settings")

# 7. mask resampling: ComfyUI MASK (B,H,W) and bare (H,W), area-averaged, clamped
m = torch.zeros(1, 4, 4); m[0, :2] = 1.0
w = k2._mask_tokens(m, (2, 2), dev)
assert torch.allclose(w, torch.tensor([1.0, 1.0, 0.0, 0.0]))
assert torch.allclose(k2._mask_tokens(m[0], (2, 2), dev), w), "(H,W) masks handled"
half = torch.zeros(1, 4, 4); half[0, 1] = 1.0        # one of two rows per token
assert torch.allclose(k2._mask_tokens(half, (2, 2), dev), torch.tensor([.5, .5, 0., 0.]))
assert k2._mask_tokens(None, (2, 2), dev) is None
print("7 ok  mask -> soft token weights")

# 8. template: empty names = training-matched, names inject exactly one {} slot
E = k2.Krea2TwoCharacterEncode
t = E._template("", "", "")
assert t.count(E.VISION) == 2 and t.count("{}") == 1
assert t == ("<|im_start|>system\n" + E.DEFAULT_SYSTEM + "<|im_end|>\n<|im_start|>user\n"
             + E.VISION + E.VISION + "{}<|im_end|>\n<|im_start|>assistant\n")
t2 = E._template("look closely at faces", "Ada", "Ben")
assert t2.count("{}") == 1 and "This is Ada." in t2 and "This is Ben." in t2
assert "look closely at faces" in t2 and E.DEFAULT_SYSTEM not in t2
t3 = E._template("", "Ada", "")
assert "This is Ada." in t3 and "the second character" in t3
assert E._template("", "A{x}", "B}").count("{}") == 1, "braces in names must not add slots"
print("8 ok  grounded-encode template")
print("\nall checks passed")
