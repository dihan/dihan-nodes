# Node Documentation

## FaceLineMask Node

The FaceLineMask node is designed to create intelligent masks based on face positions in an image. It's particularly useful for creating split-face effects or transitional masks between two faces.

### Technical Details

**Category:** FaceAnalysis

### Inputs

| Input | Type | Description |
|-------|------|-------------|
| analysis_models | ANALYSIS_MODELS | Face detection and analysis models |
| image | IMAGE | Input image to analyze |
| width | INT | Output mask width (1-8192, default: 512) |
| height | INT | Output mask height (1-8192, default: 512) |
| feather_amount | INT | Edge feathering amount (0-100, default: 0) |
| mask_side | BOOLEAN | Which side to mask (True=Right, False=Left) |

### Outputs

| Output | Type | Description |
|--------|------|-------------|
| MASK | MASK | Generated mask tensor |

### Features

1. **Automatic Face Detection**
   - Uses provided analysis models to detect faces in the image
   - Works with any number of faces (requires at least 2 for mask generation)

2. **Intelligent Mask Generation**
   - Creates a diagonal line between face centers
   - Automatically adjusts to face positions
   - Supports any angle between faces

3. **Customizable Output**
   - Adjustable output resolution
   - Feathering for smooth transitions
   - Selectable mask side

### Usage Tips

1. **Resolution Matching**
   - Set width and height to match your target image size
   - Supports high resolutions up to 8192x8192

2. **Feathering**
   - Use feather_amount to create soft transitions
   - Higher values create smoother blends
   - Recommended range: 10-50 for most uses

3. **Face Detection**
   - Ensure faces are clearly visible
   - Works best with front-facing or slightly angled faces
   - Requires at least 2 faces in the image

### Example Workflows

1. **Basic Split Face Effect**
   ```
   Image -> FaceLineMask -> Mask Output
   ```

2. **Feathered Transition**
   ```
   Image -> FaceLineMask (feather_amount=30) -> Blend
   ```

### Common Issues and Solutions

1. **No Mask Generated**
   - Check if faces are clearly visible
   - Ensure at least 2 faces are present
   - Verify analysis_models connection

2. **Mask in Wrong Direction**
   - Toggle mask_side parameter
   - Adjust face positions in input image

3. **Blurry Edges**
   - Reduce feather_amount
   - Increase image resolution

### Integration with Other Nodes

The FaceLineMask node works well with:
- Image blending nodes
- Mask processing nodes
- InstantID nodes (particularly [ComfyUI InstantID](https://github.com/cubiq/ComfyUI_InstantID))
- Other face processing workflows

### Performance Considerations

- Processing time scales with image resolution
- Face detection is the most compute-intensive step
- Feathering can add slight processing overhead 