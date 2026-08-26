# Tests

Offline checks for the Krea2 two-character nodes. They stub out ComfyUI, so they run
anywhere `torch` and `einops` are importable — no ComfyUI, no model weights, no GPU.

```bash
python3 tests/test_krea2_bias.py       # attention-bias math and the encode template
python3 tests/test_krea2_forward.py    # sequence layout / RoPE frames, against a stub DiT
KREA2EDIT_PATH=/path/to/comfyui-krea2edit python3 tests/test_krea2_geometry.py
```

`test_krea2_geometry.py` compares our vendored reference geometry against
[comfyui-krea2edit](https://github.com/lbouaraba/comfyui-krea2edit) pixel-for-pixel;
it skips itself without `KREA2EDIT_PATH`. Re-run it whenever that pack ships a
geometry change — train/infer geometry has to stay byte-identical to what the LoRA
was trained on.

The FaceAnalysis nodes have no tests here; they need the FaceAnalysis fork's models.
