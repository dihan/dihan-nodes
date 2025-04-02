from .face_detection_marker import FaceDetectionMarker
from .face_line_mask import FaceLineMask

NODE_CLASS_MAPPINGS = {
    "FaceDetectionMarker": FaceDetectionMarker,
    "FaceLineMask": FaceLineMask
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FaceDetectionMarker": "Face Detection Marker",
    "FaceLineMask": "Face Line Mask"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS'] 