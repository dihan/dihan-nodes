"""Krea2 two-character identity nodes.

The krea2_edit LoRA takes its reference blocks in a training order of
`[scene, subject]`, and the two slots are not interchangeable. The first is
*reproduced* (near-pixel scene preservation); the last is *re-rendered* into
that scene, which is the path a likeness actually has to survive and the reason
the model card's `ref_boost` dial (this file's `identity_b`) exists at all —
`~4` is its documented strong-likeness value, `1.0` is the dial switched off.
Feeding two *people* into those slots works (that is what these nodes are for),
but the asymmetry stays: the two dials do not sit on the same scale.

These nodes keep the reference geometry of comfyui-krea2edit (Apache-2.0,
https://github.com/lbouaraba/comfyui-krea2edit, vendored from v1.2.5 — the
crop/fit pixel path and the stride-1 RoPE placement are load-bearing and must
stay byte-identical to what the LoRA was trained on) and rebuild only the
attention-bias layer around a two-character workflow:

  * both characters get their own face mask (upstream masks the last ref only)
  * masks are soft — a feathered mask ramps the boost instead of hard-gating it
  * each character can be routed to a REGION of the output canvas, so A stays on
    the left and B on the right instead of the two faces averaging into one
  * the two reference blocks can be blinded to each other, which stops them
    blending before the target ever reads them

Every one of those levers is off/neutral by default: with nothing but the two
images wired, this node produces the same sequence as the upstream patch node
with source_image/source_image_b.

Wiring:
    LoadImage(A) --\
    LoadImage(B) ----> Krea2TwoCharacterPatch(model, vae, target_latent) -> KSampler
    UNETLoader -> LoraLoaderModelOnly --/
    EmptySD3LatentImage -> KSampler.latent_image (and -> target_latent)
    Krea2TwoCharacterEncode(clip, prompt, A, B) -> KSampler.positive
"""
import math

import torch
import torch.nn.functional as F
from einops import rearrange

import comfy.patcher_extension
import comfy.utils
import comfy.ldm.common_dit
from comfy.ldm.flux.layers import timestep_embedding

LOG = "[dihan-nodes/krea2-2char]"

# additive logit floor for "must not attend". -inf would NaN a fully-masked row;
# nothing here masks a whole row, but -1e4 is safe in fp16 (max ~65504) too.
NEG = -10000.0

# attributes _two_char_forward reaches into on the SingleStreamDiT
_DIT_ATTRS = ("patch", "channels", "tdim", "blocks", "first", "last", "tmlp",
              "tproj", "txtfusion", "txtmlp", "pe_embedder", "_unpack_context")


def _to_4d(v):
    """(B,C,T,H,W) -> (B*T,C,H,W); pass 4D through. Images use T=1."""
    if v.ndim == 5:
        b, c, t, h, w = v.shape
        return v.reshape(b * t, c, h, w)
    return v


def _imgids(bs, frame, h_, w_, device):
    ids = torch.zeros(h_, w_, 3, device=device, dtype=torch.float32)
    ids[..., 0] = frame
    ids[..., 1] = torch.arange(h_, device=device, dtype=torch.float32)[:, None]
    ids[..., 2] = torch.arange(w_, device=device, dtype=torch.float32)[None, :]
    return ids.reshape(1, h_ * w_, 3).repeat(bs, 1, 1)


def _imgids_offset(bs, frame, gh, gw, th, tw, device):
    """Stride-1 integer positions at a centered (fractional) offset — for `fit`
    refs, whose pixels were already resampled to the target's grid density."""
    off_h, off_w = max(0.0, (th - gh) / 2), max(0.0, (tw - gw) / 2)
    ids = torch.zeros(gh, gw, 3, device=device, dtype=torch.float32)
    ids[..., 0] = frame
    ids[..., 1] = (torch.arange(gh, device=device, dtype=torch.float32) + off_h)[:, None]
    ids[..., 2] = (torch.arange(gw, device=device, dtype=torch.float32) + off_w)[None, :]
    return ids.reshape(1, gh * gw, 3).repeat(bs, 1, 1)


def _fit_encode_image(image, vae, H, W, cache, key, fit_mode="fit"):
    """Pixel-space reference prep, then VAE-encode. Cached per (slot, resolution)
    so the encode runs once, not once per sampling step.

    Geometry is vendored from comfyui-krea2edit v1.2.5 and must not be "cleaned
    up": the /16 snap, the crop-to-grid and the fit/crop split all mirror the
    trainer's _fit_prep. Train/infer geometry has to match to the pixel or the
    reference band seams against the target.
    """
    key = key + (fit_mode,)
    if key in cache:
        return cache[key]
    px_h, px_w = H * 8, W * 8
    img = image.movedim(-1, 1)  # B,H,W,C -> B,C,H,W
    ih, iw = img.shape[-2:]
    if fit_mode == "fit":
        sc = min(px_h / ih, px_w / iw)
        # near-matched AR: fill the target grid exactly with a minimal center-crop,
        # so `fit` and `crop` agree and no target edge column is left without a
        # reference correspondence to copy from.
        CROP_TOL = 0.08
        if ih * sc >= px_h * (1 - CROP_TOL) and iw * sc >= px_w * (1 - CROP_TOL):
            s = max(px_h / ih, px_w / iw)
            ch, cw = min(ih, int(round(px_h / s))), min(iw, int(round(px_w / s)))
            y0, x0 = (ih - ch) // 2, (iw - cw) // 2
            img = img[..., y0:y0 + ch, x0:x0 + cw]
            nh, nw = px_h, px_w
        else:
            nh = min(max(16, int(ih * sc) // 16 * 16), max(16, px_h // 16 * 16))
            nw = min(max(16, int(iw * sc) // 16 * 16), max(16, px_w // 16 * 16))
            # crop so the fitted axis lands on the /16 grid at scale sc exactly:
            # resizing ih*sc -> floor16 would squash content by up to 15px and the
            # misregistration shows up as a doubled band at the reference edge.
            ch2, cw2 = min(ih, max(1, int(round(nh / sc)))), min(iw, max(1, int(round(nw / sc))))
            y0, x0 = (ih - ch2) // 2, (iw - cw2) // 2
            img = img[..., y0:y0 + ch2, x0:x0 + cw2]
        img = F.interpolate(img.float(), size=(nh, nw), mode="bicubic", antialias=True)
    else:
        # crop (legacy): center-crop to the target AR, then resize to it exactly.
        s = max(px_h / ih, px_w / iw)
        ch, cw = min(ih, int(round(px_h / s))), min(iw, int(round(px_w / s)))
        y0, x0 = (ih - ch) // 2, (iw - cw) // 2
        img = img[..., y0:y0 + ch, x0:x0 + cw]
        img = F.interpolate(img.float(), size=(px_h, px_w), mode="bicubic", antialias=True)
    lat = vae.encode(img.movedim(1, -1)[..., :3].clamp(0, 1))
    cache[key] = lat
    return lat


def _mask_tokens(mask, grid, device):
    """ComfyUI MASK -> (grid_h*grid_w,) soft weights in [0,1], or None.

    `area` downsampling is what makes the mask soft: a token that is half inside
    the mask comes out at ~0.5 and gets half the boost, so a feathered face mask
    ramps instead of stair-stepping at the token grid.
    """
    if mask is None:
        return None
    m = mask
    if m.ndim == 2:
        m = m[None]
    m = m[:1].float()                                            # (1,H,W)
    m = F.interpolate(m[None], size=grid, mode="area")[0, 0]     # (gh,gw)
    return m.reshape(-1).clamp(0.0, 1.0).to(device)


def _identity_bias(specs, txtlen, ref_lens, tgtlen, seqlen, isolate_refs, device, dtype):
    """Additive attention-logit bias over `[text | refA | refB | target]`.

    For reference i, target token r and reference token c:

        gain(r,c) = 1 + (boost_i - 1) * region_i(r) * face_i(c)
        gain(r,c) *= (1 - excl) + excl * region_i(r)          # if a region is set
        bias(r,c)  = log(gain)

    which is exactly "multiply this reference's attention weight by `gain`,
    before renormalisation". With no masks it collapses to a scalar log(boost)
    over the whole block — upstream's behaviour. With a region mask and
    exclusivity it also *suppresses* a reference outside its own region, which
    is the part that keeps two faces from averaging into each other.

    `isolate_refs` additionally blinds the reference blocks to one another, so
    they cannot blend before the target reads them.
    """
    offs = [txtlen]
    for ln in ref_lens:
        offs.append(offs[-1] + ln)
    rows0 = offs[-1]

    active = [s for s in specs
              if s["boost"] != 1.0 or (s["region_w"] is not None and s["excl"] > 0.0)]
    if not active and not (isolate_refs and len(ref_lens) > 1):
        return None

    bias = torch.zeros(1, 1, seqlen, seqlen, device=device, dtype=dtype)
    for i, s in enumerate(specs):
        off, ln = offs[i], ref_lens[i]
        b, face_w, region_w, excl = s["boost"], s["face_w"], s["region_w"], s["excl"]
        gated = region_w is not None and excl > 0.0
        if b == 1.0 and not gated:
            continue
        if face_w is None and region_w is None:
            bias[:, :, rows0:, off:off + ln] = math.log(max(b, 1e-4))
            continue
        fvec = torch.ones(ln, device=device) if face_w is None else face_w
        rvec = torch.ones(tgtlen, device=device) if region_w is None else region_w
        gain = rvec[:, None] * fvec[None, :]        # (tgtlen, ln)
        gain.mul_(b - 1.0).add_(1.0)
        if gated:
            gain.mul_(((1.0 - excl) + excl * rvec)[:, None])
        bias[0, 0, rows0:, off:off + ln] = gain.clamp_(min=1e-4).log_().to(dtype)

    if isolate_refs:
        for i in range(len(ref_lens)):
            for j in range(len(ref_lens)):
                if i != j:
                    bias[:, :, offs[i]:offs[i] + ref_lens[i],
                         offs[j]:offs[j] + ref_lens[j]] = NEG
    return bias


def _two_char_forward(m, x, timesteps, context, src_latents, transformer_options,
                      boosts, face_masks, region_masks, excl, isolate_refs, pos_mode):
    """Krea2 SingleStreamDiT._forward with the reference blocks prepended.

    Sequence is `[text | ref0(frame=1) | ref1(frame=2) | target(frame=0)]`; only
    the target tokens come back out.
    """
    patch = m.patch

    temporal = x.ndim == 5
    if temporal:
        b5, c5, t5, h5, w5 = x.shape
    x = _to_4d(x)
    bs, c, H_orig, W_orig = x.shape

    x = comfy.ldm.common_dit.pad_to_patch_size(x, (patch, patch), padding_mode="replicate")
    H, W = x.shape[-2], x.shape[-1]
    h_, w_ = H // patch, W // patch

    srcs = []
    for sl in src_latents:
        src = _to_4d(sl).to(x.device, x.dtype)
        if src.shape[0] != bs:
            src = src[:1].expand(bs, *src.shape[1:])
        srcs.append(comfy.ldm.common_dit.pad_to_patch_size(src, (patch, patch), padding_mode="replicate"))
    src_grids = [(s.shape[-2] // patch, s.shape[-1] // patch) for s in srcs]

    context = m._unpack_context(context)

    tgt_img = m.first(rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch))
    src_imgs = [m.first(rearrange(s, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch))
                for s in srcs]

    t = m.tmlp(timestep_embedding(timesteps, m.tdim).unsqueeze(1).to(tgt_img.dtype))
    tvec = m.tproj(t)

    context = m.txtfusion(context, mask=None, transformer_options=transformer_options)
    context = m.txtmlp(context)

    txtlen, tgtlen = context.shape[1], tgt_img.shape[1]
    ref_lens = [si.shape[1] for si in src_imgs]
    srclen = sum(ref_lens)
    combined = torch.cat([context] + src_imgs + [tgt_img], dim=1)
    device = combined.device

    if pos_mode == "stride1":
        ref_ids = [_imgids_offset(bs, i + 1, gh, gw, h_, w_, device)
                   for i, (gh, gw) in enumerate(src_grids)]
    else:
        ref_ids = [_imgids(bs, i + 1, gh, gw, device) for i, (gh, gw) in enumerate(src_grids)]
    pos = torch.cat(
        [torch.zeros(bs, txtlen, 3, device=device, dtype=torch.float32)]   # text @ 0
        + ref_ids
        + [_imgids(bs, 0, h_, w_, device)],                                # target frame=0
        dim=1)
    freqs = m.pe_embedder(pos)

    specs = [{
        "boost": boosts[i],
        "face_w": _mask_tokens(face_masks[i], src_grids[i], device),
        "region_w": _mask_tokens(region_masks[i], (h_, w_), device),
        "excl": excl,
    } for i in range(len(src_imgs))]
    attn_bias = _identity_bias(specs, txtlen, ref_lens, tgtlen, combined.shape[1],
                               isolate_refs, device, combined.dtype)

    for block in m.blocks:
        combined = block(combined, tvec, freqs, attn_bias, transformer_options=transformer_options)

    final = m.last(combined, t)
    out = final[:, txtlen + srclen: txtlen + srclen + tgtlen, :]
    out = rearrange(out, "b (h w) (c ph pw) -> b c (h ph) (w pw)",
                    h=h_, w=w_, ph=patch, pw=patch, c=m.channels)
    out = out[:, :, :H_orig, :W_orig]
    if temporal:
        out = out.reshape(b5, t5, m.channels, H_orig, W_orig).movedim(1, 2)
    return out


class Krea2TwoCharacterPatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Krea2 UNET with the krea2_edit / krea2 identity LoRA already applied (LoraLoaderModelOnly)"}),
                "vae": ("VAE", {"tooltip": "used to encode both characters at the exact output resolution — this pixel-space path is what keeps references sharp when the reference and output resolutions differ"}),
                "character_a": ("IMAGE", {"tooltip": "first character -> reference frame 1 (the LoRA's 'scene' slot). The scene slot is reproduced near-literally, so A's likeness largely arrives on its own"}),
                "character_b": ("IMAGE", {"tooltip": "second character -> reference frame 2 (the LoRA's 'subject' slot). The subject is RE-RENDERED into the scene rather than copied, so B's likeness is the one that needs identity_b turned up — the model card's strong-likeness value for this slot is ~4"}),
                "identity_a": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1000.0, "step": 0.01, "round": 0.001,
                                         "tooltip": "multiplies target->character_a attention. Upstream's 'ref_boost_a'. 1.0 = off, >1 pulls the output harder toward A's appearance, <1 loosens it. The model card gives no baseline for this slot because the scene ref is normally left alone; for a second CHARACTER, sweep it up alongside identity_b"}),
                "identity_b": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1000.0, "step": 0.01, "round": 0.001,
                                         "tooltip": "same dial for character_b. This is upstream's 'ref_boost', the fidelity dial the LoRA's model card documents: ~4 is the strong-likeness starting point (the author's own example workflow ships at 4) and >10 starts breaking. 1.0 is OFF, not a baseline — if B's likeness is weak, this is almost always why"}),
                "fit_mode": (["fit", "crop (legacy)"], {"default": "fit",
                              "tooltip": "how a character image fits a mismatched output aspect ratio: fit = resample to the target grid at a centered offset (matches how the LoRA was trained — use this); crop (legacy) = center-crop to the output AR then resize (v1/v1.1 weights only)"}),
            },
            "optional": {
                "target_latent": ("LATENT", {"tooltip": "RECOMMENDED: wire the SAME latent you feed KSampler.latent_image. Lets both characters be VAE-encoded here, before sampling starts, instead of on the first step where the VAE can evict part of the diffusion model"}),
                "face_mask_a": ("MASK", {"tooltip": "region of character_a's IMAGE that identity_a applies to, e.g. the face. Soft/feathered masks ramp the boost. Empty = the whole image"}),
                "face_mask_b": ("MASK", {"tooltip": "same for character_b's image"}),
                "region_a": ("MASK", {"tooltip": "region of the OUTPUT canvas that character_a is allowed to drive, e.g. the left half. Empty = the whole canvas"}),
                "region_b": ("MASK", {"tooltip": "same for character_b, e.g. the right half"}),
                "region_exclusivity": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05, "round": 0.001,
                                                 "tooltip": "how hard each character is suppressed OUTSIDE its own region mask. 0 = regions only amplify (no suppression); 1 = a character is fully cut off outside its region. Needs region_a/region_b wired. Try 0.5-0.8 when the two faces keep blending or swapping"}),
                "isolate_references": ("BOOLEAN", {"default": False,
                                                   "tooltip": "blind the two reference blocks to each other so they cannot mix before the target reads them. Off by default because full cross-reference attention is what the LoRA was trained with; turn it on when the two faces merge into an average of the pair"}),
                "swap_reference_order": ("BOOLEAN", {"default": False,
                                                     "tooltip": "put character_b in frame 1 and character_a in frame 2, i.e. swap which character sits in the copied 'scene' slot and which is re-rendered as the 'subject'. The dials keep following their own character (identity_a stays with character_a). Swap the images on the encode node to match"}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "dihan-nodes/krea2"
    DESCRIPTION = ("Krea2 in-context edit patch aimed at holding TWO character identities in one "
                   "output: per-character face masks, per-character output regions, and optional "
                   "reference isolation.")

    def patch(self, model, vae, character_a, character_b, identity_a=1.0, identity_b=1.0,
              fit_mode="fit", target_latent=None, face_mask_a=None, face_mask_b=None,
              region_a=None, region_b=None, region_exclusivity=0.0,
              isolate_references=False, swap_reference_order=False, **_future):
        if _future:
            print(f"{LOG} WARNING: workflow provides inputs this node version does not know "
                  f"({', '.join(sorted(_future))}). Update dihan-nodes and restart ComfyUI. "
                  f"Continuing without them.", flush=True)

        dm = getattr(model.model, "diffusion_model", None)
        missing = [a for a in _DIT_ATTRS if not hasattr(dm, a)]
        if missing:
            raise RuntimeError(
                f"{LOG} this node only works on a Krea2 (SingleStreamDiT) model — the model "
                f"wired in is a {type(dm).__name__} and is missing {missing}. Load the Krea2 "
                f"UNET with UNETLoader and apply the krea2 identity-edit LoRA to it.")

        m = model.clone()
        mm = model.model
        px_cache = {}
        state = {"announced": False}

        # slot order: index 0 -> RoPE frame 1 (copied 'scene'), index 1 -> frame 2
        # (re-rendered 'subject'); matches upstream's [ref_boost_a..., ref_boost]
        slots = [("b", character_b, identity_b, face_mask_b, region_b),
                 ("a", character_a, identity_a, face_mask_a, region_a)] if swap_reference_order else \
                [("a", character_a, identity_a, face_mask_a, region_a),
                 ("b", character_b, identity_b, face_mask_b, region_b)]
        boosts = [s[2] for s in slots]
        face_masks = [s[3] for s in slots]
        region_masks = [s[4] for s in slots]

        if region_exclusivity > 0.0 and region_a is None and region_b is None:
            print(f"{LOG} WARNING: region_exclusivity={region_exclusivity} has NO EFFECT — "
                  f"it needs region_a and/or region_b connected.", flush=True)

        # An all-zero MASK is the easy mistake here (a LoadImage whose mask was never
        # painted hands one over), and it fails silently in opposite directions
        # depending on which socket it landed on.
        for label, msk in (("face_mask_a", face_mask_a), ("face_mask_b", face_mask_b)):
            if msk is not None and float(msk.max()) <= 0.0:
                print(f"{LOG} WARNING: '{label}' is empty (all zero), which disables "
                      f"identity_{label[-1]} entirely — every reference token gets a boost of "
                      f"1.0. Paint the mask, or disconnect it to boost the whole image.",
                      flush=True)
        for label, msk in (("region_a", region_a), ("region_b", region_b)):
            if msk is not None and float(msk.max()) <= 0.0:
                print(f"{LOG} WARNING: '{label}' is empty (all zero) — character "
                      f"{label[-1].upper()} has no region to drive"
                      + (f", and region_exclusivity={region_exclusivity} will suppress it "
                         f"across the whole canvas." if region_exclusivity > 0 else "."),
                      flush=True)

        # Pre-encode outside the sampling window. vae.encode -> load_models_gpu ->
        # free_memory(keep_loaded=[]) partially unloads whatever is resident, the
        # diffusion model included, and nothing re-expands it mid-run.
        primed = None
        if target_latent is not None:
            Hh, Ww = target_latent["samples"].shape[-2], target_latent["samples"].shape[-1]
            print(f"{LOG} pre-encoding both characters at target {Hh * 8}x{Ww * 8}px "
                  f"(fit_mode={fit_mode})", flush=True)
            for name, img, _b, _f, _r in slots:
                _fit_encode_image(img, vae, Hh, Ww, px_cache, (name, Hh, Ww), fit_mode)
            primed = (Hh, Ww)
        else:
            print(f"{LOG} NOTE: connect 'target_latent' (the same latent that feeds "
                  f"KSampler.latent_image) to encode the characters here instead of on the "
                  f"first sampling step.", flush=True)

        def wrapper(executor, x, timesteps, context, *wargs, **kwargs):
            # ComfyUI signature drift: transformer_options is the trailing dict either
            # way, and any native ref_latents are ignored (we supply our own path).
            transformer_options = kwargs.pop("transformer_options", None)
            if transformer_options is None:
                transformer_options = {}
                for a in reversed(wargs):
                    if isinstance(a, dict):
                        transformer_options = a
                        break
            xx = _to_4d(x)
            Hh, Ww = xx.shape[-2], xx.shape[-1]
            if not state["announced"]:
                state["announced"] = True
                order = "B,A" if swap_reference_order else "A,B"
                print(f"{LOG} active — refs [{order}] boosts {boosts} "
                      f"fit_mode={fit_mode} excl={region_exclusivity} "
                      f"isolate={isolate_references}", flush=True)
                if primed is not None and primed != (Hh, Ww):
                    print(f"{LOG} WARNING: 'target_latent' is {primed[0] * 8}x{primed[1] * 8}px but "
                          f"sampling is at {Hh * 8}x{Ww * 8}px — the pre-encode is unused and the "
                          f"VAE will run mid-sampling. Wire the SAME latent that feeds KSampler.",
                          flush=True)
            src_latents = [
                mm.process_latent_in(_fit_encode_image(img, vae, Hh, Ww, px_cache,
                                                       (name, Hh, Ww), fit_mode))
                for name, img, _b, _f, _r in slots]
            return _two_char_forward(
                executor.class_obj, x, timesteps, context, src_latents, transformer_options,
                boosts=boosts, face_masks=face_masks, region_masks=region_masks,
                excl=region_exclusivity, isolate_refs=isolate_references,
                pos_mode=("stride1" if fit_mode == "fit" else "anchor"))

        to = m.model_options.setdefault("transformer_options", {})
        comfy.patcher_extension.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, "krea2_two_character", wrapper, to)
        return (m,)


class Krea2TwoCharacterEncode:
    """Image-grounded instruction encode for two characters.

    The LoRA is trained with the instruction encoded *together* with the
    reference images through Qwen3-VL, so a text-only encode runs with half the
    recipe missing: the VAE reference tokens carry appearance, this carries who
    is who. Naming the characters binds "Ada hugs Ben" to specific references
    instead of leaving the model to guess which face goes where.
    """
    DEFAULT_SYSTEM = (
        "Describe the image by detailing the color, shape, size, "
        "texture, quantity, text, spatial relationships of the objects and background:"
    )
    VISION = "<|vision_start|><|image_pad|><|vision_end|>"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "prompt": ("STRING", {"multiline": True, "default": "",
                                      "tooltip": "the edit instruction, e.g. 'Ada and Ben shaking hands in a sunlit office'. Say where each character goes — the region masks on the patch node and the prompt should agree"}),
                "character_a": ("IMAGE", {"tooltip": "same image you wired to character_a on the patch node, in the same order"}),
                "character_b": ("IMAGE", {"tooltip": "same image you wired to character_b on the patch node"}),
            },
            "optional": {
                "name_a": ("STRING", {"default": "", "tooltip": "optional name for character_a, inserted next to its image so the prompt can refer to it by name. Empty on BOTH names = the exact training-matched template"}),
                "name_b": ("STRING", {"default": "", "tooltip": "optional name for character_b"}),
                "grounding_px": ("INT", {"default": 768, "min": 0, "max": 4096, "step": 64,
                                         "tooltip": "cap on the longest side fed to Qwen3-VL; 0 = native. The LoRA trained on 384-768px, so 640-768 is in-distribution"}),
                "system_prompt": ("STRING", {"multiline": True, "default": "",
                                             "tooltip": "advanced: override the grounding system prompt (empty = training default). Steers what the vision encoder attends to — adding a facial-detail instruction here can help identity"}),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = "dihan-nodes/krea2"
    DESCRIPTION = ("Encodes the instruction grounded on BOTH character images (training-matched "
                   "semantic path), optionally naming each one so the prompt can address them.")

    @classmethod
    def _template(cls, system_prompt, name_a, name_b):
        sp = system_prompt.strip() or cls.DEFAULT_SYSTEM
        # braces would collide with the {} slot clip.tokenize formats the prompt into
        a, b = name_a.strip().replace("{", "").replace("}", ""), name_b.strip().replace("{", "").replace("}", "")
        if a or b:
            body = (f"{cls.VISION}This is {a or 'the first character'}. "
                    f"{cls.VISION}This is {b or 'the second character'}. " + "{}")
        else:
            body = cls.VISION + cls.VISION + "{}"
        return ("<|im_start|>system\n" + sp + "<|im_end|>\n<|im_start|>user\n"
                + body + "<|im_end|>\n<|im_start|>assistant\n")

    def _prep(self, image, grounding_px):
        samples = image.movedim(-1, 1)  # B,H,W,C -> B,C,H,W
        h, w = samples.shape[2], samples.shape[3]
        if grounding_px and max(h, w) > grounding_px:
            s = grounding_px / max(h, w)
            samples = comfy.utils.common_upscale(samples, round(w * s), round(h * s), "area", "disabled")
        return samples.movedim(1, -1)[:, :, :, :3]

    def encode(self, clip, prompt, character_a, character_b, name_a="", name_b="",
               grounding_px=768, system_prompt="", **_future):
        if _future:
            print(f"{LOG} WARNING: workflow provides inputs this node version does not know "
                  f"({', '.join(sorted(_future))}). Update dihan-nodes and restart ComfyUI.",
                  flush=True)
        imgs = [self._prep(character_a, grounding_px), self._prep(character_b, grounding_px)]
        tokens = clip.tokenize(prompt, images=imgs,
                               llama_template=self._template(system_prompt, name_a, name_b))
        return (clip.encode_from_tokens_scheduled(tokens),)


NODE_CLASS_MAPPINGS = {
    "Krea2TwoCharacterPatch": Krea2TwoCharacterPatch,
    "Krea2TwoCharacterEncode": Krea2TwoCharacterEncode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Krea2TwoCharacterPatch": "Krea2 Two-Character Identity (patch)",
    "Krea2TwoCharacterEncode": "Krea2 Two-Character Encode",
}
