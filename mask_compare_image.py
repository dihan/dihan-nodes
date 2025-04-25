import torch
import numpy as np
import torchvision.transforms as T
from PIL import Image
from nodes import PreviewImage

class ImageOverlayCompare(PreviewImage):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
                "opacity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO"
            }
        }

    RETURN_TYPES = ()  # No output types since we're only showing preview
    FUNCTION = "compare_images"
    CATEGORY = "image/overlay"

    def compare_images(self, image_a, image_b, opacity, filename_prefix="image_overlay.", prompt=None, extra_pnginfo=None):
        # Take first image if batch is provided
        img_a = image_a[0] if len(image_a.shape) == 4 else image_a
        img_b = image_b[0] if len(image_b.shape) == 4 else image_b
        
        # Convert tensors to PIL Images
        img_a_pil = T.ToPILImage()(img_a.permute(2, 0, 1)).convert('RGB')
        img_b_pil = T.ToPILImage()(img_b.permute(2, 0, 1)).convert('RGB')
        
        # Resize image_b to match image_a dimensions if they don't match
        if img_b_pil.size != img_a_pil.size:
            img_b_pil = img_b_pil.resize(img_a_pil.size, Image.Resampling.NEAREST)
        
        # Create a grayscale overlay from image_b
        overlay = Image.new('RGBA', img_a_pil.size, (255, 255, 255, 0))
        
        # Convert image_b to grayscale and use it as alpha
        img_b_gray = img_b_pil.convert('L')
        overlay_array = np.array(overlay)
        img_b_array = np.array(img_b_gray)
        overlay_array[:, :, 3] = img_b_array * opacity
        overlay = Image.fromarray(overlay_array)
        
        # Convert the main image to RGBA
        img_a_rgba = img_a_pil.convert('RGBA')
        
        # Create a new image with the overlay
        result = Image.alpha_composite(img_a_rgba, overlay)
        
        # Convert back to RGB
        result = result.convert('RGB')
        
        # Convert to tensor
        result_tensor = T.ToTensor()(result)
        result_tensor = result_tensor.permute(1, 2, 0)
        
        # Add batch dimension if needed
        if len(image_a.shape) == 4:
            result_tensor = result_tensor.unsqueeze(0)
        
        # Save the image using PreviewImage functionality
        saved = self.save_images(result_tensor, filename_prefix, prompt, extra_pnginfo)
        
        return saved 