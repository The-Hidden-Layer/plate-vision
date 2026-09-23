"""Bounded classical crop processing; the original crop is never modified.

These are explicit inference profiles, separate from checkpoint preprocessing.
The default exactly preserves the trained bilinear RGB letterbox. Filters run
on at most 192x48 pixels, before adding neutral padding, and never infer glyphs.
"""

from PIL import Image, ImageFilter, ImageOps

VERSION = "classical-lpr-v1"
SIZE = (192, 48)
PADDING = (127, 127, 127)
PROFILES = (
    "none",
    "bicubic",
    "lanczos",
    "small_bicubic",
    "unsharp",
    "clahe",
    "bilateral",
    "bicubic_unsharp",
)


def metadata(profile="none"):
    if profile not in PROFILES:
        raise ValueError(f"unknown LPR enhancement: {profile!r}; choose from {PROFILES}")
    return {"profile": profile, "version": VERSION}


def prepare_image(image: Image.Image, profile="none") -> Image.Image:
    metadata(profile)
    rgb = image.convert("RGB")
    if not rgb.width or not rgb.height:
        raise ValueError("LPR requires a nonempty crop")
    scale = min(SIZE[0] / rgb.width, SIZE[1] / rgb.height)
    fitted = (round(rgb.width * scale), round(rgb.height * scale))
    if 0 in fitted:
        # False detections can be one pixel wide: avoid a zero-sized Pillow resize.
        rgb = rgb.resize(tuple(max(1, n) for n in fitted), Image.Resampling.BILINEAR)
    if profile == "none" or (
        profile == "small_bicubic" and (rgb.width >= SIZE[0] or rgb.height >= SIZE[1])
    ):
        return ImageOps.pad(rgb, SIZE, method=Image.Resampling.BILINEAR, color=PADDING)

    method = Image.Resampling.BILINEAR
    if profile in {"bicubic", "small_bicubic", "bicubic_unsharp"}:
        method = Image.Resampling.BICUBIC
    elif profile == "lanczos":
        method = Image.Resampling.LANCZOS
    # A single aspect-preserving resize also bounds filtering cost for large crops.
    content = ImageOps.contain(rgb, SIZE, method=method)
    if profile in {"unsharp", "bicubic_unsharp"}:
        content = content.filter(ImageFilter.UnsharpMask(radius=0.8, percent=40, threshold=2))
    elif profile in {"clahe", "bilateral"}:
        import cv2
        import numpy as np

        pixels = np.asarray(content)
        if profile == "clahe":
            lab = cv2.cvtColor(pixels, cv2.COLOR_RGB2LAB)
            luminance = lab[:, :, 0]
            equalized = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 2)).apply(luminance)
            # Blend gently; preserve chroma and avoid thresholding away Persian dots.
            lab[:, :, 0] = cv2.addWeighted(luminance, 0.5, equalized, 0.5, 0)
            pixels = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            pixels = cv2.bilateralFilter(pixels, d=5, sigmaColor=20, sigmaSpace=2)
        content = Image.fromarray(pixels)
    # Content already fits exactly on one axis; this only adds untouched padding.
    return ImageOps.pad(content, SIZE, method=Image.Resampling.NEAREST, color=PADDING)
