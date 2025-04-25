from .face_detection_marker import FaceDetectionMarker
from .face_line_mask import FaceLineMask
from .face_gender_detect import FaceGenderDetect
from .mask_compare_image import ImageOverlayCompare

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

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS'] 