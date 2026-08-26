# Node Documentation

This pack registers six nodes.

| Class | Display name | Category | File |
|-------|--------------|----------|------|
| [`FaceLineMask`](#facelinemask) | Face Line Mask | `FaceAnalysis` | `face_line_mask.py` |
| [`FaceDetectionMarker`](#facedetectionmarker) | Face Detection Marker | `FaceAnalysis` | `face_detection_marker.py` |
| [`FaceGenderDetect`](#facegenderdetect) | Face Gender Detect | `FaceAnalysis` | `face_gender_detect.py` |
| [`ImageOverlayCompare`](#imageoverlaycompare) | Image Overlay Compare | `image/overlay` | `mask_compare_image.py` |
| [`Krea2TwoCharacterPatch`](KREA2_TWO_CHARACTER.md) | Krea2 Two-Character Identity (patch) | `dihan-nodes/krea2` | `krea2_two_character.py` |
| [`Krea2TwoCharacterEncode`](KREA2_TWO_CHARACTER.md) | Krea2 Two-Character Encode | `dihan-nodes/krea2` | `krea2_two_character.py` |

The three `FaceAnalysis` nodes require an `ANALYSIS_MODELS` input supplied by the
[forked ComfyUI_FaceAnalysis_Advanced](https://github.com/dihan/ComfyUI_FaceAnalysis_Advanced).
They rely on these methods of that object:

| Method | Used by |
|--------|---------|
| `get_bbox(image, padding, padding_percent)` | FaceLineMask, FaceDetectionMarker |
| `get_gender(image, x, y, w, h)` | FaceLineMask (`auto_detect_gender`) |
| `get_gender_locations(image)` | FaceGenderDetect |

`get_gender()` and `get_gender_locations()` exist only in the fork — the upstream
FaceAnalysis package will not work with FaceLineMask's gender mode or with
FaceGenderDetect.

`ImageOverlayCompare` has no such dependency and works with any images.

The two `krea2` nodes are independent of FaceAnalysis entirely — they need a Krea 2
model with the identity-edit LoRA, and are documented in
[KREA2_TWO_CHARACTER.md](KREA2_TWO_CHARACTER.md).

The pack also serves a mobile status dashboard at `/status`; that is a web route, not
a node, and is documented in the [README](../README.md#mobile-status-page).

---

## FaceLineMask

Creates a mask divided along the line between two detected faces. Built for
split-face effects and transitional masks.

**Category:** `FaceAnalysis` · **Function:** `create_mask`

### Inputs

| Input | Type | Default | Range | Description |
|-------|------|---------|-------|-------------|
| `analysis_models` | ANALYSIS_MODELS | — | — | Face detection and analysis models |
| `image` | IMAGE | — | — | Input image to analyze |
| `width` | INT | 512 | 1–8192 | Output mask width |
| `height` | INT | 512 | 1–8192 | Output mask height |
| `feather_amount` | INT | 0 | 0–100 | Gaussian blur radius on the mask edge |
| `mask_side` | BOOLEAN | True | Right / Left | Which side of the dividing line is filled |
| `auto_detect_faces` | BOOLEAN | False | On / Off | Off returns a fully white mask without detecting |
| `auto_detect_gender` | BOOLEAN | False | On / Off | Derive `mask_side` from the first face's gender |

### Outputs

| Output | Type | Description |
|--------|------|-------------|
| `MASK` | MASK | Mask tensor, shape `[1, height, width]`, values 0.0–1.0 |
| `Multiple Faces (Bool)` | BOOLEAN | True when the detector found 2 or more faces |

### How it works

1. **Bypass check.** If `auto_detect_faces` is Off, the node immediately returns a
   fully white `[1, height, width]` mask and skips detection entirely. This is the
   default state.
2. **Detection.** Otherwise `get_bbox(image, 0, 0.0)` is called and
   `Multiple Faces (Bool)` is set from whether 2+ faces came back.
3. **Centres.** The centres of the first two faces are computed and scaled from the
   input image's dimensions into `width` × `height` space.
4. **Optional gender override.** With `auto_detect_gender` on, `get_gender()` runs on
   both faces and `mask_side` is replaced: Right if face 1 is female, Left if male. A
   failure here is swallowed and your manual `mask_side` is kept.
5. **Dividing line.** The angle between the two centres is measured, rotated 90°, and
   a signed-distance test against that perpendicular line through the midpoint fills
   one half of the mask with 255. Any diagonal arrangement works.
6. **Feathering.** If `feather_amount > 0`, a Gaussian blur is applied with the radius
   scaled by the average of the width and height scale factors, so feathering stays
   visually consistent as you change the output resolution.
7. **Normalise.** The result is divided by 255 and returned as a float tensor.

### Behaviour to be aware of

- `auto_detect_faces` defaults to **Off**, meaning the out-of-the-box output is a
  fully white mask. If your mask looks like it is doing nothing, this is why.
- With `auto_detect_faces` on but **fewer than two faces** found, no dividing line is
  drawn and the mask stays fully black.
- Only the **first image** of a batch is processed.
- `IS_CHANGED` returns `NaN`, so this node re-executes on every run and is never
  served from ComfyUI's cache.

### Usage tips

- **Resolution:** set `width`/`height` to match the image the mask will be applied to.
- **Feathering:** 10–50 suits most blends; higher values soften the seam further.
- **Detection:** works best with front-facing or slightly angled, clearly visible
  faces.
- **Branching:** route `Multiple Faces (Bool)` into a switch to fall back gracefully
  when only one face is present.

### Example workflows

```
Image -> FaceLineMask (auto_detect_faces=On) -> Mask Output
Image -> FaceLineMask (feather_amount=30)    -> Blend
```

### Common issues

| Symptom | Cause / fix |
|---------|-------------|
| Mask is entirely white | `auto_detect_faces` is Off — turn it On |
| Mask is entirely black | Fewer than two faces detected; check with FaceDetectionMarker |
| Mask on the wrong side | Toggle `mask_side`, or turn off `auto_detect_gender` if it is overriding you |
| Edges too blurry | Lower `feather_amount`, or raise the output resolution |
| No gender effect | The upstream FaceAnalysis lacks `get_gender()`; use the fork |

### Integration

Works well with image blending nodes, mask processing nodes,
[ComfyUI InstantID](https://github.com/cubiq/ComfyUI_InstantID), and the other nodes
in this pack.

### Performance

Processing time scales with image resolution. Face detection is the most
compute-intensive step; feathering adds slight overhead.

---

## FaceDetectionMarker

Draws a bounding box around every detected face. Primarily a diagnostic node.

**Category:** `FaceAnalysis` · **Function:** `mark_faces`

### Inputs

| Input | Type | Default | Range | Description |
|-------|------|---------|-------|-------------|
| `analysis_models` | ANALYSIS_MODELS | — | — | Face detection and analysis models |
| `image` | IMAGE | — | — | Input image to analyze |
| `marker_color` | COMBO | `red` | red, green, blue, yellow, white | Box outline colour |
| `line_width` | INT | 2 | 1–10 | Box outline thickness in pixels |
| `padding` | INT | 0 | 0–4096 | Fixed pixel padding added to each box |
| `padding_percent` | FLOAT | 0.0 | 0.0–2.0, step 0.05 | Proportional padding added to each box |

### Outputs

| Output | Type | Description |
|--------|------|-------------|
| `IMAGE` | IMAGE | Input batch with boxes drawn on |

### Notes

- Processes the **entire batch** and returns a batch of the same length.
- Draws **every** detected face, unlike FaceLineMask which only uses the first two.
- `padding` and `padding_percent` are forwarded straight to `get_bbox()`, so the boxes
  you see are the padded boxes any downstream crop would use — this is the point of
  the node. Tune padding here, then reuse the values elsewhere.
- Output is normalised to 0.0–1.0 float and stacked back into a batch tensor.

---

## FaceGenderDetect

Annotates each detected face with its predicted gender.

**Category:** `FaceAnalysis` · **Function:** `detect_gender`

### Inputs

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| `analysis_models` | ANALYSIS_MODELS | — | Face detection and analysis models |
| `image` | IMAGE | — | Input image to analyze |
| `generate_image_overlay` | BOOLEAN | True | Draw labelled boxes; False passes the image through unchanged |

### Outputs

| Output | Type | Description |
|--------|------|-------------|
| `IMAGE` | IMAGE | Annotated batch (or the untouched input when the overlay is off) |

### Rendering

| Gender | Box colour | Label |
|--------|-----------|-------|
| Male | Blue `(0, 0, 255)` | `MALE` |
| Female | Magenta `(255, 0, 255)` | `FEMALE` |

Boxes are 2px; the label sits 10px above the box.

### Notes

- Every detection is also printed to the ComfyUI console in cyan with its
  coordinates — visible in the terminal, or in the status page's log pane.
- Font selection tries, in order, DejaVuSans-Bold (Linux), Arial (Windows), and
  Helvetica (macOS) at 32px, falling back to PIL's built-in bitmap font. On the
  fallback the label will be very small relative to a high-resolution image.
- Drawing happens in RGBA; the output is sliced back to 3 channels before return.
- Raises `No faces detected in images.` if the output list ends up empty.
- Depends on `get_gender_locations()`, which only the fork provides.

---

## ImageOverlayCompare

Composites `image_b` over `image_a` as a translucent white layer whose alpha comes
from image_b's luminance. Intended for visually checking a mask against its source.

**Category:** `image/overlay` · **Function:** `compare_images` · Extends `PreviewImage`

### Inputs

| Input | Type | Default | Range | Description |
|-------|------|---------|-------|-------------|
| `image_a` | IMAGE | — | — | Base image |
| `image_b` | IMAGE | — | — | Overlay; read as grayscale and used as alpha |
| `opacity` | FLOAT | 0.5 | 0.0–1.0, step 0.01 | Overlay strength |

Hidden inputs `prompt` and `extra_pnginfo` are supplied by ComfyUI and embedded in the
saved preview.

### Outputs

**None.** `RETURN_TYPES` is empty. The node subclasses `PreviewImage` and renders the
composite inside itself; nothing is passed downstream. Wire it as a terminal node.

### How it works

1. The first image of each batch is taken.
2. If `image_b`'s dimensions differ from `image_a`'s, it is resized to match using
   **nearest-neighbour** sampling — deliberate, so hard mask edges are not smoothed
   into gradients by the resize.
3. `image_b` is converted to grayscale and written into the alpha channel of a solid
   white RGBA layer, scaled by `opacity`.
4. That layer is alpha-composited over `image_a` and flattened back to RGB.
5. The result is saved through `PreviewImage.save_images()` with the filename prefix
   `image_overlay.`

### Notes

- Bright regions of `image_b` appear as white haze; dark regions stay transparent.
  Feeding a MASK-derived image into `image_b` shows exactly what it covers.
- Because the overlay is always white, this compares *coverage*, not colour. It is not
  a general-purpose A/B image blender.
- Only the first image of a batch is used.
