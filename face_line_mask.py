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
            line_length = math.sqrt(width**2 + height**2) * 2  # Double the length to ensure coverage
            mid_x = (face1_center[0] + face2_center[0]) // 2
            mid_y = (face1_center[1] + face2_center[1]) // 2
            
            # Calculate endpoints of the line
            x1 = mid_x - line_length * math.cos(perp_angle)
            y1 = mid_y - line_length * math.sin(perp_angle)
            x2 = mid_x + line_length * math.cos(perp_angle)
            y2 = mid_y + line_length * math.sin(perp_angle)
            
            # Create a polygon for filling
            # Calculate a point on one side of the line to determine which side to fill
            test_x = mid_x + math.cos(angle) * 10  # Point slightly offset in direction of second face
            test_y = mid_y + math.sin(angle) * 10
            
            # Create points for the polygon
            if mask_side:  # Right side
                points = [
                    (x1, y1),
                    (x2, y2),
                    (width + line_length, height + line_length),
                    (width + line_length, -line_length),
                    (x1, y1)
                ]
            else:  # Left side
                points = [
                    (x1, y1),
                    (x2, y2),
                    (-line_length, height + line_length),
                    (-line_length, -line_length),
                    (x1, y1)
                ]
            
            # Draw the filled polygon
            draw = ImageDraw.Draw(mask)
            draw.polygon(points, fill=255)
            
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