"""Native RGB interaction crops for the experimental Agent; no automatic locator.

Geometry follows the existing tomato-openpi offline preview: tight interaction
box -> 1.5x square context -> native crop -> encoder-specific resize.
"""
import math
import numpy as np


def square_context(box, width, height, scale=1.5):
    a = np.asarray(box, dtype=float)
    if (a.shape != (4,) or not np.isfinite(a).all()
            or not 0 <= a[0] < a[2] <= width or not 0 <= a[1] < a[3] <= height
            or not math.isfinite(scale) or scale < 1):
        raise ValueError('Finite native xyxy interaction box and scale>=1 required')
    x1, y1, x2, y2 = a
    required = max(math.ceil(x2)-math.floor(x1), math.ceil(y2)-math.floor(y1), 2)
    side = max(required, math.ceil(max(x2-x1, y2-y1)*scale))
    side = min(side, min(width, height) if required <= min(width, height) else max(width, height))

    def fit(center, low, high, length):
        lower = max(0, math.ceil(high-side)) if side <= length else length-side
        upper = min(length-side, math.floor(low)) if side <= length else 0
        if lower > upper:
            raise ValueError('Cannot contain interaction region')
        return min(max(math.floor(center-side/2), lower), upper)

    left, top = fit((x1+x2)/2, x1, x2, width), fit((y1+y2)/2, y1, y2, height)
    return left, top, left+side, top+side


def crop_rgb(image, interaction_box=None, *, valid=True, size=256, scale=1.5):
    """Return clean pixels and geometry. Invalid ROI uses the FRESH full frame.

    Default 256 is the Agent contract, not the old 224 preview specification.
    No box overlays, action labels, or predicted outcomes enter these pixels.
    """
    import cv2
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 1:
        raise ValueError('Native uint8 RGB image required')
    if type(size) is not int or size < 1 or type(valid) is not bool:
        raise ValueError('Positive output size and explicit ROI validity required')
    h, w = image.shape[:2]
    box = square_context(interaction_box if valid else (0, 0, w, h), w, h, scale)
    left, top, right, bottom = box
    native = np.zeros((right-left, right-left, 3), dtype=np.uint8)
    x1, y1, x2, y2 = max(0, left), max(0, top), min(w, right), min(h, bottom)
    native[y1-top:y2-top, x1-left:x2-left] = image[y1:y2, x1:x2]
    pixels = cv2.resize(native, (size, size), interpolation=cv2.INTER_AREA)
    return dict(rgb=pixels, crop_box=box, valid=valid, source_size=(w, h),
                fallback=None if valid else 'fresh_full_view_square_padding')


def encode_four_views(encoder, global_rgb, local_rgb):
    """Encode real synchronous [B,3/1,16,256,256,3] RGB, no B0 mutations."""
    from tools.tomato_phase_model import encode_clips
    if (global_rgb.dtype != np.uint8 or global_rgb.ndim != 6
            or global_rgb.shape[1:] != (3, 16, 256, 256, 3)
            or local_rgb.dtype != np.uint8
            or local_rgb.shape != (len(global_rgb), 16, 256, 256, 3)
            or not len(global_rgb)):
        raise ValueError('Synchronous three global +one local uint8 video batches required')
    rgb = np.concatenate((global_rgb, local_rgb[:, None]), axis=1)
    return encode_clips(encoder, rgb.reshape(-1, 16, 256, 256, 3)).reshape(-1, 4, 1024)
