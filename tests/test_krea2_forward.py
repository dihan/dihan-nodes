"""Round-trips _two_char_forward through a stub SingleStreamDiT.

first/last/blocks are all identity, so a correct implementation must hand back
exactly the target latent it was given: any patchify/unpatchify or token-slice
mistake shows up as a mismatch, and the recorded bias/pos tell us the sequence
was laid out as [text | refA | refB | target] with RoPE frames 1, 2, 0.
"""
import importlib.util
import math
import os
import sys
import types

import torch

_NODE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "krea2_two_character.py")

for name in ("comfy", "comfy.patcher_extension", "comfy.utils", "comfy.ldm",
             "comfy.ldm.common_dit", "comfy.ldm.flux", "comfy.ldm.flux.layers"):
    sys.modules.setdefault(name, types.ModuleType(name))

def pad_to_patch_size(t, ps, padding_mode="replicate"):
    h, w = t.shape[-2:]
    ph = (ps[0] - h % ps[0]) % ps[0]
    pw = (ps[1] - w % ps[1]) % ps[1]
    return t if (ph or pw) == 0 else torch.nn.functional.pad(t, (0, pw, 0, ph), mode=padding_mode)

sys.modules["comfy.ldm.common_dit"].pad_to_patch_size = pad_to_patch_size
sys.modules["comfy"].ldm = sys.modules["comfy.ldm"]
sys.modules["comfy.ldm"].common_dit = sys.modules["comfy.ldm.common_dit"]
sys.modules["comfy.ldm"].flux = sys.modules["comfy.ldm.flux"]
sys.modules["comfy.ldm.flux"].layers = sys.modules["comfy.ldm.flux.layers"]
sys.modules["comfy.ldm.flux.layers"].timestep_embedding = lambda t, dim, **k: torch.zeros(t.shape[0], dim)

spec = importlib.util.spec_from_file_location(
    "k2", _NODE)
k2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(k2)

PATCH, CH = 2, 4
DIM = CH * PATCH * PATCH   # keeps first/last as plain identities

class Block:
    def __init__(self, log): self.log = log
    def __call__(self, x, tvec, freqs, bias, transformer_options=None):
        self.log.append((tuple(x.shape), None if bias is None else tuple(bias.shape)))
        return x

class StubDiT:
    def __init__(self):
        self.patch, self.channels, self.tdim = PATCH, CH, 8
        self.seen = []
        self.blocks = [Block(self.seen)]
        self.pos_seen = []
    def _unpack_context(self, c): return c
    def first(self, x): return x
    def last(self, x, t): return x
    def tmlp(self, t): return t
    def tproj(self, t): return t
    def txtfusion(self, c, mask=None, transformer_options=None): return c
    def txtmlp(self, c): return c
    def pe_embedder(self, pos): self.pos_seen.append(pos); return pos

def run(x, srcs, txtlen=3, **kw):
    m = StubDiT()
    ctx = torch.zeros(x.shape[0], txtlen, DIM)
    kw.setdefault("boosts", [1.0, 1.0]); kw.setdefault("face_masks", [None, None])
    kw.setdefault("region_masks", [None, None]); kw.setdefault("excl", 0.0)
    kw.setdefault("isolate_refs", False); kw.setdefault("pos_mode", "stride1")
    out = k2._two_char_forward(m, x, torch.zeros(x.shape[0]), ctx, srcs, {}, **kw)
    return m, out

# 1. identity round-trip, equal-size refs
x = torch.randn(2, CH, 8, 12)
srcs = [torch.randn(1, CH, 8, 12), torch.randn(1, CH, 8, 12)]
m, out = run(x, srcs)
assert out.shape == x.shape and torch.allclose(out, x, atol=1e-6), "target tokens mis-sliced"
seq, bshape = m.seen[0]
assert seq == (2, 3 + 24 + 24 + 24, DIM), seq          # [text | refA | refB | target]
assert bshape is None, "neutral settings must not allocate a bias"
print("1 ok  round-trip + sequence layout", seq)

# 2. RoPE frames are text=0, refs=1,2, target=0 and refs are centered
pos = m.pos_seen[0]
assert (pos[0, :3, 0] == 0).all()
assert (pos[0, 3:27, 0] == 1).all() and (pos[0, 27:51, 0] == 2).all()
assert (pos[0, 51:, 0] == 0).all()
print("2 ok  RoPE frame indices 0 | 1 | 2 | 0")

# 3. a smaller `fit` ref is centered on the target grid (stride 1, fractional offset)
small = torch.randn(1, CH, 4, 8)                        # grid 2x4 inside target 4x6
m2, out2 = run(x, [small, srcs[1]])
assert torch.allclose(out2, x, atol=1e-6)
p = m2.pos_seen[0][0, 3:3 + 8]                          # refA tokens
rows, cols = p[:, 1].unique(), p[:, 2].unique()
assert torch.allclose(rows, torch.tensor([1.0, 2.0])), rows      # (4-2)/2 = 1.0
assert torch.allclose(cols, torch.tensor([1.0, 2.0, 3.0, 4.0])), cols  # (6-4)/2 = 1.0
print("3 ok  stride-1 centered ref placement", rows.tolist(), cols.tolist())

# 4. anchor mode (crop/legacy) pins refs at the origin instead
m3, _ = run(x, [small, srcs[1]], pos_mode="anchor")
p = m3.pos_seen[0][0, 3:3 + 8]
assert p[:, 1].min() == 0 and p[:, 2].min() == 0
print("4 ok  anchor mode pins to origin")

# 5. odd sizes get padded to the patch grid and cropped back on the way out
xo = torch.randn(1, CH, 7, 9)
m4, out4 = run(xo, [torch.randn(1, CH, 7, 9), torch.randn(1, CH, 8, 8)])
assert out4.shape == xo.shape and torch.allclose(out4, xo, atol=1e-6)
print("5 ok  odd latent sizes pad/crop cleanly")

# 6. 5D (B,C,T,H,W) latents survive the flatten/restore
x5 = torch.randn(2, CH, 1, 8, 12)
m5, out5 = run(x5, srcs)
assert out5.shape == x5.shape and torch.allclose(out5, x5, atol=1e-6)
print("6 ok  5D temporal latents round-trip")

# 7. a single-image batch reference is broadcast to a batched target
m6, out6 = run(torch.randn(3, CH, 8, 12), srcs)
assert out6.shape[0] == 3
print("7 ok  ref broadcast over batch")

# 8. non-neutral settings do allocate the bias, at full sequence size
m7, _ = run(x, srcs, boosts=[1.4, 1.0])
_, bshape = m7.seen[0]
assert bshape == (1, 1, 75, 75), bshape
print("8 ok  bias allocated at [L,L] when a dial is off-neutral")
print("\nall checks passed")
