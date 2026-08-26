# Krea2 Two-Character Identity

Two nodes for holding **two people's identities** in one Krea 2 in-context edit:

| Class | Display name | Category | File |
|-------|--------------|----------|------|
| `Krea2TwoCharacterPatch` | Krea2 Two-Character Identity (patch) | `dihan-nodes/krea2` | `krea2_two_character.py` |
| `Krea2TwoCharacterEncode` | Krea2 Two-Character Encode | `dihan-nodes/krea2` | `krea2_two_character.py` |

---

## Why these exist

The krea2 identity-edit LoRA takes **two** reference blocks, and it was trained on the
order `[scene, subject]` — the *last* reference is the one it treats as "the thing to
preserve". Feeding two *people* into those slots works, but the asymmetry doesn't go
away: the second slot pulls harder on the output. That is why, on the upstream
`Krea2EditModelPatch`, balanced identities usually need `ref_boost` (the last ref, i.e.
`source_image_b`) set **lower** than `ref_boost_a` (the first ref) — the second
character is already winning before you touch a dial.

These nodes keep the reference geometry identical to
[comfyui-krea2edit](https://github.com/lbouaraba/comfyui-krea2edit) v1.2.5 — the
crop/fit pixel path and the stride-1 RoPE placement are load-bearing and have to match
what the LoRA was trained on to the pixel — and rebuild only the **attention-bias
layer** around the two-character case:

- **Per-character face masks.** Upstream can mask the last reference only; here each
  character has its own, so you can pull identity from A's face *and* B's face.
- **Soft masks.** A feathered mask ramps the boost instead of hard-gating it at the
  token grid, so mask edges don't print into the result.
- **Output-canvas regions.** Each character can be routed to a region of the *output*
  — A on the left, B on the right — instead of both competing over every pixel.
- **Reference isolation.** The two reference blocks can be blinded to each other, so
  they can't average into a single face before the target ever reads them.
- **Order swap.** One toggle moves a character into the favoured last slot, to A/B
  test which one benefits from it without rewiring the graph.

Every one of those is off or neutral by default. With just the two images wired, this
node builds exactly the same sequence as the upstream patch node with
`source_image` / `source_image_b`.

## Requirements

- ComfyUI with Krea 2 support (the node needs `comfy.patcher_extension`).
- The Krea 2 UNET, plus the krea2 identity-edit LoRA applied to it with
  `LoraLoaderModelOnly`. Weights: <https://huggingface.co/conradlocke/krea2-identity-edit>.
- The native Krea 2 CLIP (a `qwen3vl` text encoder **with** its vision tower — the
  encode node feeds images through it).

You do **not** need `comfyui-krea2edit` installed. If you do have it, don't stack its
patch node and this one on the same model — each prepends its own reference blocks.

## Wiring

```
UNETLoader ──► LoraLoaderModelOnly ──┐
                                     ▼
LoadImage (character A) ────────► Krea2TwoCharacterPatch ──► KSampler.model
LoadImage (character B) ────────►   ▲          ▲
VAELoader ──────────────────────────┘          │
EmptySD3LatentImage ──┬── target_latent ───────┘
                      └──────────────────────────► KSampler.latent_image

CLIPLoader ──► Krea2TwoCharacterEncode(prompt, A, B) ──► KSampler.positive
            └► Krea2TwoCharacterEncode("",     A, B) ──► KSampler.negative   (CFG > 1 only)
```

`target_latent` is optional but wire it: it lets both characters be VAE-encoded at
node-execution time instead of on the first sampling step, where `vae.encode` can make
ComfyUI evict part of the resident diffusion model and leave the rest of the run
streaming weights from CPU.

At CFG > 1, ground the negative too — a second encode node with an **empty** prompt and
the same two images. That is what the LoRA's unconditional was trained on.

---

## `Krea2TwoCharacterPatch`

### Required inputs

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | `MODEL` | — | Krea 2 UNET with the identity-edit LoRA already applied |
| `vae` | `VAE` | — | encodes both characters at the exact output resolution (the pixel-space path — this is what keeps references sharp when reference and output resolutions differ) |
| `character_a` | `IMAGE` | — | first character → RoPE frame 1 (the LoRA's "scene" slot) |
| `character_b` | `IMAGE` | — | second character → RoPE frame 2 (the "subject" slot, structurally favoured) |
| `identity_a` | `FLOAT` | `1.0` | multiplies target→A attention. `1.0` off, `>1` pulls harder toward A's appearance, `<1` loosens |
| `identity_b` | `FLOAT` | `1.0` | same for B |
| `fit_mode` | choice | `fit` | how a character image fits a mismatched output aspect ratio. `fit` resamples to the target grid at a centered offset (training-matched — use this); `crop (legacy)` center-crops to the output AR then resizes (v1/v1.1 weights only) |

### Optional inputs

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| `target_latent` | `LATENT` | — | the **same** latent you feed `KSampler.latent_image`; moves the VAE encode out of the sampling window |
| `face_mask_a` | `MASK` | — | region of **A's image** that `identity_a` applies to, e.g. the face. Soft masks ramp the boost. Empty = whole image |
| `face_mask_b` | `MASK` | — | same for B's image |
| `region_a` | `MASK` | — | region of the **output canvas** A is allowed to drive, e.g. the left half. Empty = whole canvas |
| `region_b` | `MASK` | — | same for B |
| `region_exclusivity` | `FLOAT` | `0.0` | how hard each character is suppressed *outside* its own region. `0` = regions only amplify; `1` = fully cut off outside. Needs a region mask wired |
| `isolate_references` | `BOOLEAN` | `false` | blind the two reference blocks to each other |
| `swap_reference_order` | `BOOLEAN` | `false` | put B in frame 1 and A in frame 2. `identity_a` / `face_mask_a` / `region_a` keep following `character_a` |

Output: `MODEL` — feed it to `KSampler`.

### What the dials actually do

The node builds the sequence `[text | refA | refB | target]` and adds a bias to the
attention logits. For reference *i*, target token *r* and reference token *c*:

```
gain(r,c) = 1 + (identity_i - 1) · region_i(r) · face_i(c)
gain(r,c) ·= (1 - exclusivity) + exclusivity · region_i(r)     # if a region is wired
bias(r,c)  = log gain(r,c)
```

Adding `log g` to a logit multiplies that key's attention weight by `g` before
renormalisation — so `identity_a = 2.0` means "A's reference tokens count double when
the output decides what to look like". With no masks it collapses to a single scalar
over the whole block, which is exactly upstream's `ref_boost`.

`isolate_references` is separate: it drops the refA↔refB logits to `-1e4`, so the two
references stop reading each other in every block. It is **off** by default because
full cross-reference attention is what the LoRA trained with; turning it on is
deliberately off-distribution and is the lever for "the two faces merged into an
average of the pair".

---

## `Krea2TwoCharacterEncode`

The LoRA encodes the instruction *together with* the reference images through Qwen3-VL
and taps 12 layers. A plain `CLIPTextEncode` runs with half the recipe missing: the VAE
reference tokens carry appearance, this carries **who is who**.

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| `clip` | `CLIP` | — | the native Krea 2 `qwen3vl` encoder, vision tower included |
| `prompt` | `STRING` | `""` | the instruction. Say where each character goes; the prompt and the region masks should agree |
| `character_a` | `IMAGE` | — | same image, same order, as on the patch node |
| `character_b` | `IMAGE` | — | " |
| `name_a` | `STRING` | `""` | optional name inserted next to A's image, so the prompt can address it by name |
| `name_b` | `STRING` | `""` | same for B |
| `grounding_px` | `INT` | `768` | cap on the longest side fed to the VLM; `0` = native. The LoRA trained on 384–768 px, so 640–768 is in-distribution |
| `system_prompt` | `STRING` | `""` | advanced: override the grounding system prompt (empty = training default) |

With both names empty the template is byte-identical to the training-matched two-image
template. Fill them in and the user turn becomes
`<image A> This is Ada. <image B> This is Ben. <prompt>`, which gives the model a handle
to bind "Ada hugs Ben" to specific references instead of guessing which face goes where.
Naming is a small deviation from the trained template — if it costs you quality, clear
both names.

---

## Tuning recipe

Change one thing at a time; the dials interact.

1. **Baseline.** Both identities at `1.0`, no masks. Prompt describes both characters
   and where each one is. Generate at **≤ 2 MP** — above the trained range, references
   start bleeding into the output and subjects duplicate.
2. **One identity is weak.** Raise that character's dial in `0.1`–`0.2` steps and leave
   the other at `1.0`. Because B sits in the favoured slot, the balanced pair usually
   ends up with **A above B** (e.g. A `1.4` / B `1.0`), which is the same asymmetry as
   upstream's `ref_boost < ref_boost_a`.
3. **Faces blend into one person.** Add `face_mask_a` and `face_mask_b` so the boost
   targets faces rather than clothes and background. If they still merge, turn on
   `isolate_references`.
4. **Faces swap sides, or both come out as one character.** Wire `region_a` and
   `region_b` (rough halves are enough) and set `region_exclusivity` to `0.5`–`0.8`.
   Keep the prompt consistent with the masks.
5. **Output looks pasted-together / too literal.** Both dials below `1.0` loosens the
   copy and lets the model restage the pair.
6. **Still stuck.** Flip `swap_reference_order` and re-run step 2 — it tests whether the
   *other* character wants the favoured slot. Swap the images on the encode node to match.

Prefer `euler` (or another ODE sampler) over `er_sde`: the SDE noise injection disrupts
the reference-copy channel.

## Migrating an existing krea2edit graph

Drop-in replacement for the two upstream nodes:

| `Krea2EditModelPatch` | → | `Krea2TwoCharacterPatch` |
|---|---|---|
| `source_image` (first ref / "scene") | → | `character_a` |
| `source_image_b` (last ref / "subject") | → | `character_b` |
| `ref_boost_a` (first ref) | → | `identity_a` |
| `ref_boost` (last ref) | → | `identity_b` |
| `ref_boost_mask` (last ref only) | → | `face_mask_b` (and `face_mask_a` is now available too) |
| `vae`, `fit_mode`, `target_latent` | → | same names |
| `source_latent` / `source_latent_b` | → | *not offered* — this node always takes images + `vae` |

| `Krea2EditGroundedEncode` | → | `Krea2TwoCharacterEncode` |
|---|---|---|
| `image` | → | `character_a` |
| `image_b` | → | `character_b` |
| `prompt`, `grounding_px`, `system_prompt` | → | same names |

Note the `ref_boost` ↔ `identity_b` crossover: upstream names the dials by *slot*, this
node names them by *character*.

## Cost and limits

- The attention bias is an `L × L` tensor in the model's working dtype, where
  `L = text + refA + refB + target` tokens. It is only allocated when something is
  off-neutral — with both identities at `1.0`, no region exclusivity and isolation off,
  no bias tensor is built at all and the cost is zero. At ~1 MP with two references it
  lands in the few-hundred-MB range, and it also pushes attention off the flash path,
  so expect it to be slower than a plain run. This is the same cost upstream pays
  whenever `ref_boost != 1.0`.
- Both characters are VAE-encoded once per output resolution and cached, not per step.
- Two people in one pass is more reliable than chaining two single-character edits.
  Face separation is a known weak spot of the LoRA itself; these controls push against
  it but can't fully fix it.

## Credit

The reference geometry (`_fit_encode_image`, the RoPE placement, the wrapper signature
handling) is adapted from [comfyui-krea2edit](https://github.com/lbouaraba/comfyui-krea2edit)
v1.2.5 by lbouaraba, Apache-2.0. The two-character attention layer and these nodes are
MIT, like the rest of this pack.
