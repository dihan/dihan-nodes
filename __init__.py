from .face_detection_marker import FaceDetectionMarker
from .face_line_mask import FaceLineMask
from .face_gender_detect import FaceGenderDetect
from .mask_compare_image import ImageOverlayCompare
from . import status_page
from . import mmh3_page

# Serves the mobile status dashboard at /status on ComfyUI's own port.
status_page.setup()

# Serves the MiniMax H3 Ref2VA generator page at /mmh3.
mmh3_page.setup()

NODE_CLASS_MAPPINGS = {
    "FaceDetectionMarker": FaceDetectionMarker,
    "FaceLineMask": FaceLineMask,
    "FaceGenderDetect": FaceGenderDetect,
    "ImageOverlayCompare": ImageOverlayCompare
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FaceDetectionMarker": "Face Detection Marker",
    "FaceLineMask": "Face Line Mask",
    "FaceGenderDetect": "Face Gender Detect",
    "ImageOverlayCompare": "Image Overlay Compare"
}

# The Krea2 nodes reach into comfy internals (patcher_extension, ldm.common_dit,
# the flux timestep embedding). On a ComfyUI too old to have them, load the rest
# of the pack anyway instead of taking every node down with the import.
try:
    from . import krea2_two_character
    NODE_CLASS_MAPPINGS.update(krea2_two_character.NODE_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(krea2_two_character.NODE_DISPLAY_NAME_MAPPINGS)
except Exception as e:
    print(f"[dihan-nodes] Krea2 two-character nodes not loaded: {type(e).__name__}: {e}", flush=True)

# MiniMax H3 prompt writer (any LLM + the H3 guide + a checker). Guarded the same
# way, so a problem there never takes the other nodes down with it.
try:
    from . import mmh3_prompt
    NODE_CLASS_MAPPINGS.update(mmh3_prompt.NODE_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(mmh3_prompt.NODE_DISPLAY_NAME_MAPPINGS)
except Exception as e:
    print(f"[dihan-nodes] MMH3 prompt nodes not loaded: {type(e).__name__}: {e}", flush=True)

# Frontend extensions only (the prompt nodes' text preview). The /status and /mmh3
# pages in web/ are served by their own routes, not from here.
WEB_DIRECTORY = "./web/js"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']
