# Tests

Offline checks for the Krea2 two-character nodes. They stub out ComfyUI, so they run
anywhere `torch` and `einops` are importable — no ComfyUI, no model weights, no GPU.

```bash
python3 tests/test_krea2_bias.py       # attention-bias math and the encode template
python3 tests/test_krea2_forward.py    # sequence layout / RoPE frames, against a stub DiT
python3 tests/test_krea2_workflow.py   # the shipped workflow vs. the node definitions
KREA2EDIT_PATH=/path/to/comfyui-krea2edit python3 tests/test_krea2_geometry.py
```

`test_krea2_geometry.py` compares our vendored reference geometry against
[comfyui-krea2edit](https://github.com/lbouaraba/comfyui-krea2edit) pixel-for-pixel;
it skips itself without `KREA2EDIT_PATH`. Re-run it whenever that pack ships a
geometry change — train/infer geometry has to stay byte-identical to what the LoRA
was trained on.

`test_krea2_workflow.py` checks `workflow/Krea2 Two-Character Identity.json` against
`INPUT_TYPES` — socket names and types, widget order and ranges, link bookkeeping in
both directions, and the wiring the docs promise. Regenerate the graph with
`python3 workflow/_build_krea2_two_character.py "workflow/Krea2 Two-Character Identity.json"`
after changing either the builder or the node's inputs, then re-run it.

The FaceAnalysis nodes have no tests here; they need the FaceAnalysis fork's models.
