"""MiniMax H3 prompt writer: choose any LLM, give it the H3 guide (Ref2VA by default),
check the draft against the guide's hard rules, let the model fix failures, save, load.

Pure python plus numpy/Pillow for images; no extra installs. See docs/MMH3_PROMPT_WRITER.md.
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
