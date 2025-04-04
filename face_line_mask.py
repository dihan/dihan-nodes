import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter
import torchvision.transforms as T
import math

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
                "mask_side": ("BOOLEAN", {"default": True, "label_on": "Right", "label_off": "Left"}),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "create_mask"
    CATEGORY = "FaceAnalysis"

    def create_mask(self, analysis_models, image, width, height, feather_amount, mask_side):
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
            
            # Calculate the angle between the two face centers
            dx = face2_center[0] - face1_center[0]
            dy = face2_center[1] - face1_center[1]
            angle = math.atan2(dy, dx)
            
            # Calculate the perpendicular angle
            perp_angle = angle + math.pi/2
            
            # Calculate points for the dividing line
            # Make the line extend beyond the image boundaries
            line_length = math.sqrt(width**2 + height**2)
            mid_x = (face1_center[0] + face2_center[0]) // 2
            mid_y = (face1_center[1] + face2_center[1]) // 2
            
            # Calculate endpoints of the line
            x1 = mid_x - line_length * math.cos(perp_angle)
            y1 = mid_y - line_length * math.sin(perp_angle)
            x2 = mid_x + line_length * math.cos(perp_angle)
            y2 = mid_y + line_length * math.sin(perp_angle)
            
            # Create a temporary image for the line mask
            temp_mask = Image.new('L', (width, height), 0)
            temp_draw = ImageDraw.Draw(temp_mask)
            temp_draw.line([(x1, y1), (x2, y2)], fill=255, width=1)
            
            # Fill the appropriate side based on mask_side
            mask_array = np.array(temp_mask, dtype=np.uint8)
            mask_array = np.where(mask_array > 0, 255, 0).astype(np.uint8)
            
            if mask_side:
                # When True, left side is 1.0 (masked)
                # Fill the left side of the line
                for y in range(height):
                    line_points = np.where(mask_array[y] > 0)[0]
                    if len(line_points) > 0:
                        left_point = line_points[0]
                        mask_array[y, :left_point] = 255
            else:
                # When False, right side is 1.0 (masked)
                # Fill the right side of the line
                for y in range(height):
                    line_points = np.where(mask_array[y] > 0)[0]
                    if len(line_points) > 0:
                        right_point = line_points[-1]
                        mask_array[y, right_point:] = 255
            
            # Convert back to PIL Image
            mask = Image.fromarray(mask_array)
            
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