import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter
import torchvision.transforms as T

class FaceLineMask:
    MAX_RESOLUTION = 8192  # Match ComfyUI's default max resolution

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "analysis_models": ("ANALYSIS_MODELS",),
                "image": ("IMAGE",),
                "width": ("INT", {"default": 512, "min": 1, "max": s.MAX_RESOLUTION, "step": 1}),
                "height": ("INT", {"default": 512, "min": 1, "max": s.MAX_RESOLUTION, "step": 1}),
                "feather_amount": ("INT", {"default": 0, "min": 0, "max": 100}),
                "left_side_mask": ("BOOLEAN", {"default": True, "label_on": "Left", "label_off": "Right"}),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "create_mask"
    CATEGORY = "FaceAnalysis"

    def create_mask(self, analysis_models, image, width, height, feather_amount, left_side_mask):
        # Take first image if batch is provided
        i = image[0] if len(image.shape) == 4 else image
        
        # Convert tensor to PIL Image
        i = T.ToPILImage()(i.permute(2, 0, 1)).convert('RGB')
        
        # Get face detection results
        img, x, y, w, h = analysis_models.get_bbox(i, 0, 0.0)
        
        # Calculate scale factors for the new dimensions
        scale_x = width / i.size[0]
        scale_y = height / i.size[1]
        
        # Create a base mask at the target size
        mask = Image.new('L', (width, height), 0)
        draw = ImageDraw.Draw(mask)
        
        # If we have at least 2 faces, create the division mask
        if len(x) >= 2:
            # Get centers of first two faces and scale them
            face1_center = (int((x[0] + w[0]//2) * scale_x), int((y[0] + h[0]//2) * scale_y))
            face2_center = (int((x[1] + w[1]//2) * scale_x), int((y[1] + h[1]//2) * scale_y))
            
            # Calculate the midpoint between faces
            mid_x = (face1_center[0] + face2_center[0]) // 2
            
            # Fill one side with 1.0 (255 in PIL) based on left_side_mask
            if left_side_mask:
                # When True, left side is 1.0 (masked)
                draw.rectangle([0, 0, mid_x, height], fill=255)
                draw.rectangle([mid_x, 0, width, height], fill=0)
            else:
                # When False, right side is 1.0 (masked)
                draw.rectangle([0, 0, mid_x, height], fill=0)
                draw.rectangle([mid_x, 0, width, height], fill=255)
            
            # Add feathering if requested
            if feather_amount > 0:
                # Scale feather amount based on the new dimensions
                scaled_feather = int(feather_amount * (scale_x + scale_y) / 2)
                mask = mask.filter(ImageFilter.GaussianBlur(radius=scaled_feather))
        
        # Convert mask to numpy array and normalize to 0-1
        mask_array = np.array(mask).astype(np.float32) / 255.0
        
        # Convert to tensor with shape [1, height, width]
        mask_tensor = torch.from_numpy(mask_array)
        mask_tensor = mask_tensor.unsqueeze(0)  # Add batch dimension
        
        return (mask_tensor,) 