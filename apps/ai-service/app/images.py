import warnings
from io import BytesIO

from PIL import Image, ImageOps, UnidentifiedImageError


class InvalidImage(ValueError):
    status_code = 422


class UnsupportedImage(InvalidImage):
    status_code = 415


class ImageTooLarge(InvalidImage):
    status_code = 413


def decode_image(source, *, max_pixels):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(source) if isinstance(source, bytes) else source) as image:
                if image.format not in {"JPEG", "PNG", "WEBP", "BMP"}:
                    raise UnsupportedImage("supported image formats: JPEG, PNG, WEBP, BMP")
                if image.width * image.height > max_pixels:
                    raise ImageTooLarge(f"image exceeds {max_pixels} decoded pixels")
                if getattr(image, "n_frames", 1) != 1:
                    raise UnsupportedImage("animated/multi-frame images are not supported")
                return ImageOps.exif_transpose(image).convert("RGB")
    except InvalidImage:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ImageTooLarge("image exceeds the safe decoded-pixel limit") from exc
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidImage("image is corrupt or cannot be decoded") from exc
