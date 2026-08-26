from .face_detection_marker import FaceDetectionMarker
from .face_line_mask import FaceLineMask
from .face_gender_detect import FaceGenderDetect
from .mask_compare_image import ImageOverlayCompare
from . import status_page

# Serves the mobile status dashboard at /status on ComfyUI's own port.
status_page.setup()

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

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
