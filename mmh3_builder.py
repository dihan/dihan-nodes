"""Builds the MiniMax H3 Ref2VA chain as a ComfyUI API prompt.

This is the graph from the "minimax-ref2va" workflow, rebuilt from a small
settings dict so the /mmh3 page can drive it without the ComfyUI canvas:

    refs -> [Reference to Video -> Motion Context -> sample -> decode -> trim] x N
         -> join -> save low-res mp4 -> (free VRAM -> SeedVR2 -> save upscaled mp4)

Each segment pins the tail of the previous segment's sampler latent, so the
segments play as one continuous shot. The first segment can pin the tail of a
latent saved by an earlier run, which is how a chain continues across runs.

The canvas-only helpers from the original workflow (prompt stack, shot list,
switches, show-text, notes) are dropped: their job was to assemble a prompt
string, and that is done here instead. Pure python with no ComfyUI imports, so
it can be tested on its own.
"""

import math
import random

MAX_SEGMENTS = 6
MAX_REFS = 6
SEED_MAX = 2 ** 50

# Model files and node settings taken from the original workflow.
DEFAULTS = {
    "unet": "MiniMax_H3_Ref2VA_pruned_int8_convrot.safetensors",
    "clip": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
    "video_vae": "minimax_h3_video_vae_fp16.safetensors",
    "audio_vae": "minimax_h3_audio_vae_fp32.safetensors",
    "snap": 32,
    "width": 736,
    "height": 1280,
    "seconds": 8,
    "fps": 24,
    "steps": 8,
    "sampler": "euler_ancestral",
    "scheduler": "simple",
    "ref_image_size": "match",
    "ref_max_mp": 2,
    "context_length": "22",
    "audio_context_length": 22,
    "latent_folder": "h3_context",
    "load_index": 0,
    "save_index": 1,
    "save_every_segment": False,
    "prefix": "minimax-ref2va",
    "save_segments": True,
    "upscale": {
        "enabled": True,
        "resolution": 1080,
        "batch_size": 5,
        "seed": 196308778,
        "color_correction": "lab",
        "dit": "seedvr2_ema_7b-Q4_K_M.gguf",
        "vae": "ema_vae_fp16.safetensors",
        "blocks_to_swap": 36,
    },
    "loras": [
        {"name": "MMH3/minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors",
         "on": True, "strength": 1.0},
        {"name": "MMH3/ref2VA_Motion_v2.safetensors", "on": True, "strength": 0.5},
        {"name": "MMH3/M3_Unlocked_V2.safetensors", "on": False, "strength": 0.55},
    ],
}

CONTEXT_LENGTHS = ("5", "22", "39", "56")


class BuildError(ValueError):
    """Settings that cannot produce a runnable graph."""


def frame_count(seconds, fps=24):
    """Frames for a clip of `seconds`, snapped up to H3's 17k+5 grid."""
    wanted = max(5, int(round(float(seconds) * int(fps))))
    return int(math.ceil((wanted - 5) / 17.0)) * 17 + 5


SNAP_OPTIONS = (0, 8, 16, 32, 64)


def snap_dim(value, step=32):
    """Round a dimension to a multiple of `step`, the way the page does.

    Halves go away from zero (JS Math.round), not to even as python's round()
    would, so a value typed in the page and the same value rebuilt here can
    never disagree. step 0 leaves the number alone.
    """
    n = max(32, int(float(value)))
    step = int(step) if int(step) in SNAP_OPTIONS else 32
    if not step:
        return n
    return max(step, (n + step // 2) // step * step)


def assemble_prompt(blocks, shot):
    """Join the enabled shared blocks around the shot text.

    Blocks marked "after" (sound, room consistency) follow the shot, the rest
    precede it, matching the SHARED / shot / SHARED sound order of the canvas.
    """
    before, after = [], []
    for block in blocks or []:
        if not block.get("on", True):
            continue
        text = str(block.get("text", "")).strip()
        if not text:
            continue
        (after if block.get("where") == "after" else before).append(text)
    parts = before + ([shot.strip()] if shot and shot.strip() else []) + after
    return "\n\n".join(parts)


def _merged(settings):
    cfg = dict(DEFAULTS)
    cfg.update({k: v for k, v in (settings or {}).items() if v is not None})
    up = dict(DEFAULTS["upscale"])
    up.update((settings or {}).get("upscale") or {})
    cfg["upscale"] = up
    return cfg


def _mini_state(max_mp):
    # Same resize settings the workflow's Load Image Mini nodes used:
    # cap at max_mp megapixels, never crop.
    return (
        '{"version":1,"mode":"max_mp","max_mp":%s,"longest_side":1024,'
        '"scale_factor":1,"snap":32,"resample":"auto","allow_upscale":true}'
        % float(max_mp)
    )


def build(settings, rng=None):
    """Return (prompt, info) for the given settings.

    `info` carries what the page needs back: the seeds actually used (so a
    random seed can be kept) and the output layout.
    """
    rng = rng or random.Random()
    cfg = _merged(settings)

    refs = [r for r in (cfg.get("refs") or []) if r and r.get("image")]
    if not refs:
        raise BuildError("Add at least one reference image.")
    if len(refs) > MAX_REFS:
        raise BuildError("At most %d reference images." % MAX_REFS)

    segments = [s for s in (cfg.get("segments") or []) if s is not None]
    if not segments:
        raise BuildError("Add at least one segment.")
    if len(segments) > MAX_SEGMENTS:
        raise BuildError("At most %d segments per run." % MAX_SEGMENTS)

    snap = cfg.get("snap", 32)
    width, height = snap_dim(cfg["width"], snap), snap_dim(cfg["height"], snap)
    fps = int(cfg["fps"])
    length = frame_count(cfg["seconds"], fps)
    context_length = str(cfg["context_length"])
    if context_length not in CONTEXT_LENGTHS:
        raise BuildError("Context length must be one of %s." % ", ".join(CONTEXT_LENGTHS))
    steps = max(1, int(cfg["steps"]))
    prefix = str(cfg["prefix"]).strip().strip("/") or DEFAULTS["prefix"]

    p = {}

    def node(node_id, class_type, title, **inputs):
        p[node_id] = {"class_type": class_type, "inputs": inputs, "_meta": {"title": title}}
        return node_id

    # --- models -----------------------------------------------------------
    node("unet", "UNETLoader", "Load Diffusion Model",
         unet_name=cfg["unet"], weight_dtype="default")
    node("clip", "CLIPLoader", "Load CLIP",
         clip_name=cfg["clip"], type="stable_diffusion", device="default")
    node("vae", "VAELoader", "Load video VAE", vae_name=cfg["video_vae"])
    node("avae", "VAELoader", "Load audio VAE", vae_name=cfg["audio_vae"])

    model = ["unet", 0]
    for i, lora in enumerate(cfg.get("loras") or []):
        if not lora.get("on") or not lora.get("name"):
            continue
        strength = float(lora.get("strength", 1.0))
        if strength == 0:
            continue
        nid = node("lora%d" % (i + 1), "LoraLoaderModelOnly", "LoRA %s" % lora["name"],
                   model=model, lora_name=lora["name"], strength_model=strength)
        model = [nid, 0]

    # --- references -------------------------------------------------------
    ref_inputs = {}
    for i, ref in enumerate(refs):
        nid = node("ref%d" % (i + 1), "PixaromaLoadImageMini", "Picture %d" % (i + 1),
                   image=ref["image"], LoadImageMiniState=_mini_state(cfg["ref_max_mp"]))
        ref_inputs["ref_images.ref_image_%d" % i] = [nid, 0]

    # --- segments ---------------------------------------------------------
    blocks = cfg.get("blocks") or []
    save_slot = int(cfg["save_index"])
    save_each = bool(cfg.get("save_every_segment"))
    seeds, prev_latent, clips = [], None, []
    if int(cfg["load_index"]) > 0:
        prev_latent = [node("ctx_load", "MiniMaxH3MotionContextLoadLatent",
                            "Continue from saved clip %d" % int(cfg["load_index"]),
                            latent_path=cfg["latent_folder"],
                            clip_index=int(cfg["load_index"])), 0]

    for n, seg in enumerate(segments, start=1):
        text = assemble_prompt(blocks, seg.get("prompt", ""))
        if not text:
            raise BuildError("Segment %d has an empty prompt." % n)
        seed = seg.get("seed")
        if seg.get("random", True) or seed in (None, ""):
            seed = rng.randrange(SEED_MAX)
        seed = int(seed)
        seeds.append(seed)

        s = "s%d_" % n
        node(s + "r2v", "MiniMaxH3ReferenceToVideo", "Reference to Video %d" % n,
             prompt=text, width=width, height=height, length=length,
             ref_image_size=cfg["ref_image_size"],
             clip=["clip", 0], vae=["vae", 0], audio_vae=["avae", 0], **ref_inputs)

        mc = dict(context_length=context_length,
                  audio_context_length=int(cfg["audio_context_length"]),
                  conditioning=[s + "r2v", 0], vae=["vae", 0], latent=[s + "r2v", 1],
                  audio_vae=["avae", 0])
        if prev_latent is not None:
            mc["context_latent"] = prev_latent
        node(s + "ctx", "MiniMaxH3MotionContext", "Motion Context %d" % n, **mc)

        node(s + "noise", "RandomNoise", "Noise %d" % n, noise_seed=seed)
        node(s + "guider", "BasicGuider", "Guider %d" % n,
             model=model, conditioning=[s + "ctx", 0])
        node(s + "sampler", "KSamplerSelect", "Sampler %d" % n, sampler_name=cfg["sampler"])
        node(s + "sigmas", "BasicScheduler", "Scheduler %d" % n,
             scheduler=cfg["scheduler"], steps=steps, denoise=1.0, model=model)
        node(s + "sample", "SamplerCustomAdvanced", "Sample segment %d" % n,
             noise=[s + "noise", 0], guider=[s + "guider", 0], sampler=[s + "sampler", 0],
             sigmas=[s + "sigmas", 0], latent_image=[s + "r2v", 1])
        node(s + "decode", "VAEDecode", "Decode video %d" % n,
             samples=[s + "sample", 0], vae=["vae", 0])
        node(s + "adecode", "VAEDecodeAudio", "Decode audio %d" % n,
             samples=[s + "sample", 0], vae=["avae", 0])
        node(s + "trim", "MiniMaxH3MotionContextTrim", "Trim overlap %d" % n,
             trim_frames=[s + "ctx", 1], fps=fps, match_tail=True,
             images=[s + "decode", 0], audio=[s + "adecode", 0])
        if cfg["save_segments"]:
            node(s + "save", "PixaromaSaveMp4", "Save segment %d" % n,
                 fps=fps, filename_prefix="%s/clip/seg_%02d" % (prefix, n),
                 save_mode="save", trim_to_audio=False, audio_fade_ms=0,
                 video_frames=[s + "trim", 0], audio=[s + "trim", 1])

        prev_latent = [s + "sample", 0]
        clips.append(s + "trim")

        # A latent per segment costs disk but survives a stopped run: whatever
        # finished can still be continued from. Written inside the loop so it
        # lands as soon as its segment is sampled, not at the end of the run.
        if save_each and save_slot > 0:
            node(s + "ctxsave", "MiniMaxH3MotionContextSaveLatent",
                 "Save chain latent %d" % (save_slot + n - 1),
                 filename_prefix="%s/clip" % cfg["latent_folder"],
                 clip_index=save_slot + n - 1, latent=prev_latent)

    # Without per-segment saves, only the last segment is a continuation point.
    if save_slot > 0 and not save_each:
        node("ctx_save", "MiniMaxH3MotionContextSaveLatent",
             "Save chain latent %d" % save_slot,
             filename_prefix="%s/clip" % cfg["latent_folder"],
             clip_index=save_slot, latent=prev_latent)

    # --- join -------------------------------------------------------------
    video, audio = [clips[0], 0], [clips[0], 1]
    for n, clip_id in enumerate(clips[1:], start=2):
        video = [node("join_v%d" % n, "PixaromaCombine", "Join video +%d" % n,
                      any1=video, any2=[clip_id, 0]), 0]
        audio = [node("join_a%d" % n, "AudioConcat", "Join audio +%d" % n,
                      direction="after", audio1=audio, audio2=[clip_id, 1]), 0]

    node("save_low", "PixaromaSaveMp4", "Save joined mp4",
         fps=fps, filename_prefix="%s/768p/lowres_" % prefix, save_mode="save",
         trim_to_audio=False, audio_fade_ms=0, video_frames=video, audio=audio)

    # --- upscale ----------------------------------------------------------
    up = cfg["upscale"]
    if up.get("enabled"):
        node("up_free", "PixaromaFreeVram", "Free VRAM before upscale", value=video,
             FreeVramState='{"mode":"all","gc":true,"everyRun":true,'
                           '"useThreshold":false,"thresholdGb":8}')
        node("up_dit", "SeedVR2LoadDiTModel", "SeedVR2 DiT",
             model=up["dit"], device="cuda:0", blocks_to_swap=int(up["blocks_to_swap"]),
             swap_io_components=True, offload_device="cpu", cache_model=False,
             attention_mode="sdpa")
        node("up_vae", "SeedVR2LoadVAEModel", "SeedVR2 VAE",
             model=up["vae"], device="cuda:0", encode_tiled=True, encode_tile_size=512,
             encode_tile_overlap=64, decode_tiled=True, decode_tile_size=512,
             decode_tile_overlap=64, tile_debug="false", offload_device="cpu",
             cache_model=False)
        node("up_run", "SeedVR2VideoUpscaler", "SeedVR2 upscale",
             seed=int(up["seed"]), resolution=int(up["resolution"]), max_resolution=0,
             batch_size=int(up["batch_size"]), uniform_batch_size=False,
             color_correction=up["color_correction"], temporal_overlap=0,
             prepend_frames=0, input_noise_scale=0, latent_noise_scale=0,
             offload_device="cpu", enable_debug=False,
             image=["up_free", 0], dit=["up_dit", 0], vae=["up_vae", 0])
        node("save_up", "PixaromaSaveMp4", "Save upscaled mp4",
             fps=fps, filename_prefix="%s/svr2" % prefix, save_mode="save",
             trim_to_audio=False, audio_fade_ms=0,
             video_frames=["up_run", 0], audio=audio)

    trim = int(context_length)
    first_trim = trim if int(cfg["load_index"]) > 0 else 0
    delivered = length * len(segments) - trim * (len(segments) - 1) - first_trim
    info = {
        "seeds": seeds,
        "width": width,
        "height": height,
        "length": length,
        "frames": delivered,
        "seconds": round(delivered / float(fps), 2),
        # The slot a following run continues from: the last one written.
        "save_index": (save_slot + len(segments) - 1) if (save_slot and save_each) else save_slot,
        "save_every_segment": save_each,
    }
    return p, info
