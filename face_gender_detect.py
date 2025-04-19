import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageColor
import torchvision.transforms as T
import os
import traceback
import math

class FaceGenderDetect:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "analysis_models": ("ANALYSIS_MODELS", ),
                "image": ("IMAGE", ),
                "generate_image_overlay": ("BOOLEAN", { "default": True }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "detect_gender"
    CATEGORY = "FaceAnalysis"

    def detect_gender(self, analysis_models, image, generate_image_overlay=True):
        if generate_image_overlay:
            # Use default font instead of trying to load a specific font file
            try:
                # Try to load a system font
                font_paths = [
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # Linux
                    "C:/Windows/Fonts/arial.ttf",  # Windows
                    "/System/Library/Fonts/Helvetica.ttc",  # macOS
                ]
                
                font = None
                for path in font_paths:
                    if os.path.exists(path):
                        font = ImageFont.truetype(path, 32)
                        break
                
                if font is None:
                    # If no system font found, use default
                    font = ImageFont.load_default()
            except Exception as e:
                print(f"Font loading error: {e}")
                # Fall back to default font
                font = ImageFont.load_default()

        out = []
        
        for i in image:
            img = np.array(T.ToPILImage()(i.permute(2, 0, 1)).convert('RGB'))
            male_faces, female_faces = analysis_models.get_gender_locations(img)
            
            if generate_image_overlay:
                tmp = T.ToPILImage()(i.permute(2, 0, 1)).convert('RGBA')
                draw = ImageDraw.Draw(tmp)
                
                # Draw male face boxes in blue
                for bbox in male_faces:
                    x1, y1, x2, y2 = map(int, bbox)
                    draw.rectangle([x1, y1, x2, y2], outline=(0, 0, 255, 255), width=2)
                    draw.text((x1, y1-10), "MALE", fill=(0, 0, 255, 255), font=font)
                    print(f"\033[96mFace Analysis: Male face detected at [{x1}, {y1}, {x2}, {y2}]\033[0m")
                
                # Draw female face boxes in pink
                for bbox in female_faces:
                    x1, y1, x2, y2 = map(int, bbox)
                    draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 255, 255), width=2)
                    draw.text((x1, y1-10), "FEMALE", fill=(255, 0, 255, 255), font=font)
                    print(f"\033[96mFace Analysis: Female face detected at [{x1}, {y1}, {x2}, {y2}]\033[0m")
                
                out.append(T.ToTensor()(tmp).permute(1, 2, 0))
            else:
                out.append(i)

        if not out:
            raise Exception('No faces detected in images.')
    
        out = torch.stack(out)
        
        if out.shape[3] > 3:
            out = out[:, :, :, :3]

        return (out,)