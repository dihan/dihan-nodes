# ComfyUI Face Detection Marker Node

A custom node for ComfyUI that adds face detection and marking capabilities to your workflow.

![Face Line Mask Example](docs/workflow1.0.png)

## Features

- Detects faces in images
- Draws customizable bounding boxes around detected faces
- Supports multiple marker colors (red, green, blue, yellow, white)
- Adjustable line width for bounding boxes
- Configurable padding around detected faces (both absolute and percentage-based)
- Face Line Mask generation with feathering options

## Requirements

- ComfyUI
- Python 3.8+
- Dependencies listed in `requirements.txt`

## Installation

1. Clone this repository into your ComfyUI's `custom_nodes` directory
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

The node provides the following inputs:
- `analysis_models`: The face analysis models
- `image`: Input image to process
- `marker_color`: Color of the bounding box (red, green, blue, yellow, white)
- `line_width`: Width of the bounding box lines (1-10)
- `padding`: Absolute padding in pixels around detected faces
- `padding_percent`: Percentage-based padding around detected faces (0-200%)

### Example Workflows

You can find example workflows in the `workflow` directory:
- [Auto mask example 1.0.json](workflow/Auto%20mask%20example%201.0.json) - Demonstrates face line mask generation with feathering

## License

MIT License 