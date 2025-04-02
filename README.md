# ComfyUI Face Detection Marker Node

A custom node for ComfyUI that adds face detection and marking capabilities to your workflow.

## Features

- Detects faces in images
- Draws customizable bounding boxes around detected faces
- Supports multiple marker colors (red, green, blue, yellow, white)
- Adjustable line width for bounding boxes
- Configurable padding around detected faces (both absolute and percentage-based)

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

## License

MIT License 