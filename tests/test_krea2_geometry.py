"""Equivalence check: our vendored _fit_encode_image must produce byte-identical
reference pixels to comfyui-krea2edit v1.2.5. Train/infer geometry has to match
the LoRA exactly, so a refactor that shifts a crop by a pixel is a real bug.
"""
import importlib.util
import os
import sys
import types

import torch

_NODE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "krea2_two_character.py")

for name in ("comfy", "comfy.patcher_extension", "comfy.utils", "comfy.ldm",
             "comfy.ldm.common_dit", "comfy.ldm.flux", "comfy.ldm.flux.layers",
             "comfy.ldm.flux.math", "comfy.ldm.modules", "comfy.ldm.modules.attention"):
    sys.modules.setdefault(name, types.ModuleType(name))
sys.modules["comfy"].ldm = sys.modules["comfy.ldm"]
sys.modules["comfy.ldm"].common_dit = sys.modules["comfy.ldm.common_dit"]
sys.modules["comfy.ldm"].flux = sys.modules["comfy.ldm.flux"]
sys.modules["comfy.ldm"].modules = sys.modules["comfy.ldm.modules"]
sys.modules["comfy.ldm.flux"].layers = sys.modules["comfy.ldm.flux.layers"]
sys.modules["comfy.ldm.flux"].math = sys.modules["comfy.ldm.flux.math"]
sys.modules["comfy.ldm.modules"].attention = sys.modules["comfy.ldm.modules.attention"]
sys.modules["comfy.ldm.flux.layers"].timestep_embedding = lambda *a, **k: None
sys.modules["comfy.ldm.flux.math"].apply_rope = lambda *a, **k: None
sys.modules["comfy.ldm.modules.attention"].optimized_attention_masked = lambda *a, **k: None
sys.modules["comfy.ldm.common_dit"].pad_to_patch_size = lambda t, ps, **k: t

def load(mod, path):
    sp = importlib.util.spec_from_file_location(mod, path)
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m

# needs a checkout of the pack we vendored the geometry from:
#   git clone https://github.com/lbouaraba/comfyui-krea2edit /tmp/krea2edit
#   KREA2EDIT_PATH=/tmp/krea2edit python3 tests/test_krea2_geometry.py
UPSTREAM = os.environ.get("KREA2EDIT_PATH")
if not UPSTREAM or not os.path.isfile(os.path.join(UPSTREAM, "__init__.py")):
    print("SKIP: set KREA2EDIT_PATH to a comfyui-krea2edit checkout to run this test")
    raise SystemExit(0)
ours = load("k2", _NODE)
up = load("upstream", os.path.join(UPSTREAM, "__init__.py"))

class FakeVAE:              # hands back the exact pixels it was asked to encode
    def encode(self, px): return px.clone()

# reference ARs vs output ARs, incl. the near-match tolerance path and odd sizes
CASES = [
    ((512, 512), (64, 64)), ((832, 1216), (152, 104)), ((1216, 832), (104, 152)),
    ((1024, 768), (64, 64)), ((600, 900), (128, 72)),  ((1000, 1000), (74, 56)),
    ((753, 1004), (128, 96)), ((754, 1004), (128, 96)), ((480, 640), (96, 128)),
    ((2048, 1152), (72, 128)), ((300, 1200), (64, 64)),
]
checked = 0
for (ih, iw), (H, W) in CASES:
    img = torch.rand(1, ih, iw, 3)
    for mode in ("fit", "crop"):
        a = ours._fit_encode_image(img, FakeVAE(), H, W, {}, ("x", H, W), mode)
        b = up._fit_encode_image(img, FakeVAE(), H, W, {}, ("x", H, W), mode)
        assert a.shape == b.shape, f"{ih}x{iw}->{H}x{W} {mode}: {a.shape} vs {b.shape}"
        assert torch.equal(a, b), f"{ih}x{iw}->{H}x{W} {mode}: pixels differ"
        checked += 1
print(f"geometry identical to upstream v1.2.5 across {checked} case/mode pairs")

# the cache must key on slot AND resolution AND mode, or a second character
# would silently reuse the first one's encode
cache = {}
i1, i2 = torch.rand(1, 64, 64, 3), torch.rand(1, 64, 64, 3)
ours._fit_encode_image(i1, FakeVAE(), 16, 16, cache, ("a", 16, 16), "fit")
ours._fit_encode_image(i2, FakeVAE(), 16, 16, cache, ("b", 16, 16), "fit")
ours._fit_encode_image(i2, FakeVAE(), 32, 32, cache, ("b", 32, 32), "fit")
ours._fit_encode_image(i2, FakeVAE(), 16, 16, cache, ("b", 16, 16), "crop")
assert len(cache) == 4, cache.keys()
assert not torch.equal(cache[("a", 16, 16, "fit")], cache[("b", 16, 16, "fit")])
hit = ours._fit_encode_image(torch.rand(1, 64, 64, 3), FakeVAE(), 16, 16, cache, ("a", 16, 16), "fit")
assert torch.equal(hit, cache[("a", 16, 16, "fit")]), "second call must hit the cache"
print("cache keys on (slot, resolution, fit_mode) and re-serves hits")

# fit refs never exceed the target grid -> _imgids_offset's gh<=th precondition holds
for (ih, iw), (H, W) in CASES:
    lat = ours._fit_encode_image(torch.rand(1, ih, iw, 3), FakeVAE(), H, W, {}, ("x", H, W), "fit")
    ph, pw = lat.shape[1], lat.shape[2]   # FakeVAE passes pixels through as B,H,W,C
    assert ph <= H * 8 and pw <= W * 8, f"{ih}x{iw}->{H}x{W}: ref {ph}x{pw} exceeds target"
    assert ph % 16 == 0 and pw % 16 == 0, f"{ih}x{iw}->{H}x{W}: ref {ph}x{pw} off the /16 grid"
print("fit refs stay within the target grid and on the /16 snap")
print("\nall checks passed")
