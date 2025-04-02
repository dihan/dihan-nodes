import numpy as np
import torch
from PIL import Image, ImageDraw
import torchvision.transforms as T

class FaceDetectionMarker:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "analysis_models": ("ANALYSIS_MODELS",),
                "image": ("IMAGE",),
                "marker_color": (["red", "green", "blue", "yellow", "white"], {"default": "red"}),
                "line_width": ("INT", {"default": 2, "min": 1, "max": 10}),
                "padding": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
                "padding_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "mark_faces"
    CATEGORY = "FaceAnalysis"

    def mark_faces(self, analysis_models, image, marker_color, line_width, padding, padding_percent):
        out_images = []
        
        for i in image:
            # Convert tensor to PIL Image
            i = T.ToPILImage()(i.permute(2, 0, 1)).convert('RGB')
            
            # Get face detection results with padding
            img, x, y, w, h = analysis_models.get_bbox(i, padding, padding_percent)
            
            # Create a copy to draw on
            draw_image = i.copy()
            draw = ImageDraw.Draw(draw_image)
            
            # Color mapping
            color_map = {
                "red": (255, 0, 0),
                "green": (0, 255, 0),
                "blue": (0, 0, 255),
                "yellow": (255, 255, 0),
                "white": (255, 255, 255)
            }
            color = color_map[marker_color]
            
            # Draw rectangles around faces
            for face_x, face_y, face_w, face_h in zip(x, y, w, h):
                draw.rectangle(
                    [(face_x, face_y), 
                     (face_x + face_w, face_y + face_h)],
                    outline=color,
                    width=line_width
                )
            
            # Convert back to tensor
            result = np.array(draw_image).astype(np.float32) / 255.0
            result = torch.from_numpy(result)
            if len(result.shape) == 2:  # Add channel dimension if grayscale
                result = result.unsqueeze(-1)
            out_images.append(result)
        
        # Stack all images
        result = torch.stack(out_images)
        return (result,) 