"""Overlay language info onto RGB frames for debugging / GUI."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _font(size: int = 16):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def overlay_language(
    image: np.ndarray,
    instruction: str,
    current_piece: str = "",
    token: Optional[int] = None,
    bar_height: int = 56,
    text_color: Tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Draw instruction and the current-step token on an RGB image."""
    if image is None:
        return image

    base = Image.fromarray(np.asarray(image).copy()).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    width, _ = base.size
    draw.rectangle([(0, 0), (width, bar_height)], fill=(0, 0, 0, 160))

    font = _font(16)
    small = _font(14)
    draw.text((8, 4), f"Instruction: {instruction}", fill=text_color + (255,), font=font)

    piece = (current_piece or "<pad>").strip() or "<pad>"
    stream_line = f"Stream: {piece}"
    if token is not None:
        stream_line += f"  |  token_id={token}"
    draw.text((8, 30), stream_line, fill=text_color + (255,), font=small)

    composed = Image.alpha_composite(base, overlay).convert("RGB")
    return np.asarray(composed).copy()
