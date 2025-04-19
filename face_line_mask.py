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
                "auto_detect_faces": ("BOOLEAN", {"default": False, "label_on": "On", "label_off": "Off"}),
                "auto_detect_gender": ("BOOLEAN", {"default": False, "label_on": "On", "label_off": "Off"}),
            }
        }

    RETURN_TYPES = ("MASK", "Multiple Faces Bool?")
    FUNCTION = "create_mask"
    CATEGORY = "FaceAnalysis"

    def create_mask(self, analysis_models, image, width, height, feather_amount, mask_side, auto_detect_faces, auto_detect_gender):
        # Take first image if batch is provided
        i = image[0] if len(image.shape) == 4 else image
        
        # Convert tensor to PIL Image
        i = T.ToPILImage()(i.permute(2, 0, 1)).convert('RGB')
        
        # Calculate scale factors for the new dimensions
        scale_x = width / i.size[0]
        scale_y = height / i.size[1]
        
        # Create a base mask at the target size
        mask = Image.new('L', (width, height), 0)
        
        # Initialize multiple_faces_detected as False
        multiple_faces_detected = False
        
        # If auto_detect_faces is False, create a full white mask (unmask everything)
        if not auto_detect_faces:
            mask_array = np.ones((height, width), dtype=np.uint8) * 255
            mask = Image.fromarray(mask_array)
        else:
            # Get face detection results
            img, x, y, w, h = analysis_models.get_bbox(i, 0, 0.0)
            
            # Check if multiple faces were detected
            multiple_faces_detected = len(x) >= 2
            
            # If we have at least 2 faces, create the division mask
            if multiple_faces_detected:
                # Get centers of first two faces and scale them
                face1_center = (int((x[0] + w[0]//2) * scale_x), int((y[0] + h[0]//2) * scale_y))
                face2_center = (int((x[1] + w[1]//2) * scale_x), int((y[1] + h[1]//2) * scale_y))
                
                # If auto_detect_gender is enabled, determine which side to mask based on gender
                if auto_detect_gender:
                    try:
                        # Get gender predictions for both faces
                        gender1 = analysis_models.get_gender(i, x[0], y[0], w[0], h[0])
                        gender2 = analysis_models.get_gender(i, x[1], y[1], w[1], h[1])
                        
                        # Determine which side to mask based on gender
                        # If the first face is female, mask the right side (True)
                        # If the first face is male, mask the left side (False)
                        mask_side = (gender1 == "female")
                    except:
                        # If gender detection fails, keep the user-specified mask_side
                        pass
                
                # Calculate the angle between the two face centers
                dx = face2_center[0] - face1_center[0]
                dy = face2_center[1] - face1_center[1]
                angle = math.atan2(dy, dx)
                
                # Calculate the perpendicular angle for the dividing line
                perp_angle = angle + math.pi/2
                
                # Calculate the midpoint
                mid_x = (face1_center[0] + face2_center[0]) // 2
                mid_y = (face1_center[1] + face2_center[1]) // 2
                
                # Create a numpy array for the mask
                mask_array = np.zeros((height, width), dtype=np.uint8)
                
                # Create coordinate grids
                y_coords, x_coords = np.mgrid[0:height, 0:width]
                
                # Calculate the signed distance from each point to the line
                # The line equation is: (x-x0)*sin(angle) - (y-y0)*cos(angle) = 0
                # where (x0,y0) is the midpoint and angle is perpendicular to the face direction
                distances = (x_coords - mid_x) * math.sin(perp_angle) - (y_coords - mid_y) * math.cos(perp_angle)
                
                # Fill based on the distance and mask_side
                if mask_side:  # Right side
                    mask_array[distances > 0] = 255
                else:  # Left side
                    mask_array[distances < 0] = 255
                
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
        
        return (mask_tensor, multiple_faces_detected) 