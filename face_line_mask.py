import numpy as np
import torch
from PIL import Image, ImageDraw
import torchvision.transforms as T

class FaceLineMask:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "analysis_models": ("ANALYSIS_MODELS",),
                "image": ("IMAGE",),
                "line_width": ("INT", {"default": 2, "min": 1, "max": 100}),
                "invert_mask": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "create_mask"
    CATEGORY = "FaceAnalysis"

    def create_mask(self, analysis_models, image, line_width, invert_mask):
        out_masks = []
        
        for i in image:
            # Convert tensor to PIL Image
            i = T.ToPILImage()(i.permute(2, 0, 1)).convert('RGB')
            
            # Get face detection results
            img, x, y, w, h = analysis_models.get_bbox(i, 0, 0.0)
            
            # Create a black mask (will be white where we want the line)
            mask = Image.new('L', i.size, 0)
            draw = ImageDraw.Draw(mask)
            
            # If we have at least 2 faces, draw a line between them
            if len(x) >= 2:
                # Get centers of first two faces
                face1_center = (x[0] + w[0]//2, y[0] + h[0]//2)
                face2_center = (x[1] + w[1]//2, y[1] + h[1]//2)
                
                # Draw line between face centers
                draw.line([face1_center, face2_center], fill=255, width=line_width)
            
            # Convert mask to numpy array
            mask_array = np.array(mask).astype(np.float32) / 255.0
            
            # Invert the mask if requested
            if invert_mask:
                mask_array = 1.0 - mask_array
            
            # Convert to tensor
            mask_tensor = torch.from_numpy(mask_array)
            if len(mask_tensor.shape) == 2:  # Add channel dimension if needed
                mask_tensor = mask_tensor.unsqueeze(-1)
            out_masks.append(mask_tensor)
        
        # Stack all masks
        result = torch.stack(out_masks)
        return (result,) 