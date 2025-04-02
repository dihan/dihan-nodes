import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter
import torchvision.transforms as T

class FaceLineMask:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "analysis_models": ("ANALYSIS_MODELS",),
                "image": ("IMAGE",),
                "feather_amount": ("INT", {"default": 0, "min": 0, "max": 100}),
                "left_side_black": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "create_mask"
    CATEGORY = "FaceAnalysis"

    def create_mask(self, analysis_models, image, feather_amount, left_side_black):
        out_masks = []
        
        for i in image:
            # Convert tensor to PIL Image
            i = T.ToPILImage()(i.permute(2, 0, 1)).convert('RGB')
            
            # Get face detection results
            img, x, y, w, h = analysis_models.get_bbox(i, 0, 0.0)
            
            # Create a white mask
            mask = Image.new('L', i.size, 255)
            draw = ImageDraw.Draw(mask)
            
            # If we have at least 2 faces, create the division mask
            if len(x) >= 2:
                # Get centers of first two faces
                face1_center = (x[0] + w[0]//2, y[0] + h[0]//2)
                face2_center = (x[1] + w[1]//2, y[1] + h[1]//2)
                
                # Calculate the midpoint between faces
                mid_x = (face1_center[0] + face2_center[0]) // 2
                
                # Fill the left side with black
                if left_side_black:
                    draw.rectangle([0, 0, mid_x, i.size[1]], fill=0)
                else:
                    draw.rectangle([mid_x, 0, i.size[0], i.size[1]], fill=0)
                
                # Add feathering if requested
                if feather_amount > 0:
                    mask = mask.filter(ImageFilter.GaussianBlur(radius=feather_amount))
            
            # Convert mask to numpy array
            mask_array = np.array(mask).astype(np.float32) / 255.0
            
            # Convert to tensor
            mask_tensor = torch.from_numpy(mask_array)
            if len(mask_tensor.shape) == 2:  # Add channel dimension if needed
                mask_tensor = mask_tensor.unsqueeze(-1)
            out_masks.append(mask_tensor)
        
        # Stack all masks
        result = torch.stack(out_masks)
        return (result,) 