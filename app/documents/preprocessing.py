"""
Adaptive Image Preprocessing & Controlled Variant Generator.
Provides non-destructive image enhancement (orientation correction, deskew, contrast, upscaling)
designed specifically to protect decimal points, commas, currency symbols, and handwritten strokes.
"""

import math
import logging
from typing import Tuple, Dict, Any, Optional
from PIL import Image, ImageOps, ImageFilter, ExifTags
import numpy as np
import cv2

logger = logging.getLogger(__name__)


def get_exif_orientation(image: Image.Image) -> int:
    """Returns orientation angle in degrees (0, 90, 180, 270) from EXIF metadata."""
    try:
        exif = image.getexif()
        if not exif:
            return 0
        orientation_key = None
        for k, v in ExifTags.TAGS.items():
            if v == "Orientation":
                orientation_key = k
                break
        if orientation_key and orientation_key in exif:
            val = exif[orientation_key]
            if val == 3:
                return 180
            elif val == 6:
                return 270
            elif val == 8:
                return 90
    except Exception as e:
        logger.debug("EXIF orientation extraction failed: %s", e)
    return 0


def detect_text_orientation_score(cv_img_gray: np.ndarray) -> float:
    """
    Computes horizontal text line structure score.
    Natural text lines produce long horizontal bounding boxes when dilated horizontally.
    """
    _, thresh = cv2.threshold(cv_img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    morph_h = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_h)
    cnts, _ = cv2.findContours(morph_h, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ratios = [cv2.boundingRect(c)[2] / max(1, cv2.boundingRect(c)[3]) for c in cnts if cv2.contourArea(c) > 50]
    return float(sum(r for r in ratios if r > 2.0))


def detect_orientation(image: Image.Image) -> int:
    """
    Detects coarse orientation (0, 90, 180, 270 degrees).
    First checks EXIF. If absent, evaluates morphological horizontal text line score.
    """
    exif_deg = get_exif_orientation(image)
    if exif_deg != 0:
        return exif_deg

    w, h = image.size
    thumb = image.copy()
    thumb.thumbnail((500, 500), Image.Resampling.BILINEAR)
    gray = np.array(thumb.convert("L"))

    scores = {
        0: detect_text_orientation_score(gray),
        90: detect_text_orientation_score(cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)),
        180: detect_text_orientation_score(cv2.rotate(gray, cv2.ROTATE_180)),
        270: detect_text_orientation_score(cv2.rotate(gray, cv2.ROTATE_90_COUNTERCLOCKWISE)),
    }

    # If document was captured in landscape (w > h) and rotated orientations show significantly
    # stronger horizontal line structures than 0/180, select the appropriate 90/270 orientation.
    if w > h and max(scores[90], scores[270]) >= (scores[0] + 0.1) * 1.1:
        return 270 if scores[270] >= scores[90] * 0.9 else 90

    best_deg = max(scores, key=scores.get)
    return best_deg


def correct_orientation(image: Image.Image) -> Tuple[Image.Image, int]:
    """Corrects image orientation if rotated. Returns (corrected_image, angle_rotated)."""
    deg = detect_orientation(image)
    if deg != 0:
        # Counter-rotate by deg
        corrected = image.rotate(360 - deg, expand=True)
        return corrected, deg
    return image, 0


def compute_skew_angle(cv_gray: np.ndarray) -> float:
    """Estimates fine skew angle in degrees using Hough line transform in range [-15, 15]."""
    edges = cv2.Canny(cv_gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=80, maxLineGap=10)
    if lines is None:
        return 0.0

    angles = []
    for line in lines:
        coords = line[0] if len(line.shape) > 1 else line
        if len(coords) < 4:
            continue
        x1, y1, x2, y2 = coords[0], coords[1], coords[2], coords[3]
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        if dx == 0:
            continue
        angle = math.degrees(math.atan2(dy, dx))
        if -15.0 <= angle <= 15.0:
            angles.append(angle)

    if not angles:
        return 0.0

    # Median angle for robustness against noise
    median_angle = float(np.median(angles))
    return round(median_angle, 2)


def deskew_image(image: Image.Image) -> Tuple[Image.Image, float]:
    """Deskews an image if fine skew is detected."""
    cv_img = np.array(image)
    if len(cv_img.shape) == 3:
        cv_gray = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
    else:
        cv_gray = cv_img

    angle = compute_skew_angle(cv_gray)
    if abs(angle) > 0.5:
        # Rotate opposite to skew
        deskewed = image.rotate(-angle, resample=Image.Resampling.BICUBIC, expand=True)
        return deskewed, angle
    return image, 0.0


class AdaptivePreprocessor:
    """Generates evidence-driven, non-destructive preprocessing variants."""

    @staticmethod
    def variant_a_normalized(image: Image.Image) -> Image.Image:
        """Variant A (Default fast path): Correct orientation + mild deskew."""
        img, _ = correct_orientation(image)
        deskewed, _ = deskew_image(img)
        return deskewed

    @staticmethod
    def variant_b_contrast_clahe(image: Image.Image) -> Image.Image:
        """Variant B: Grayscale + CLAHE contrast enhancement for low contrast/faded ink."""
        base = AdaptivePreprocessor.variant_a_normalized(image)
        cv_img = np.array(base)
        if len(cv_img.shape) == 3:
            gray = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
        else:
            gray = cv_img

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        return Image.fromarray(enhanced).convert("RGB")

    @staticmethod
    def variant_c_adaptive_threshold(image: Image.Image) -> Image.Image:
        """
        Variant C: Soft adaptive threshold preserving thin strokes and decimal dots.
        Uses Gaussian adaptive threshold with large window to maintain continuous strokes.
        """
        base = AdaptivePreprocessor.variant_a_normalized(image)
        cv_img = np.array(base)
        if len(cv_img.shape) == 3:
            gray = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
        else:
            gray = cv_img

        # Large block size (31) and moderate C (10) preserves fine dots and commas
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
        )
        return Image.fromarray(thresh).convert("RGB")

    @staticmethod
    def variant_d_upscaled_sharpen(image: Image.Image, scale_factor: float = 1.5) -> Image.Image:
        """
        Variant D: Bicubic upscaling + mild unsharp mask for low-resolution or fine-print scans.
        """
        base = AdaptivePreprocessor.variant_a_normalized(image)
        new_w = int(base.width * scale_factor)
        new_h = int(base.height * scale_factor)
        upscaled = base.resize((new_w, new_h), Image.Resampling.BICUBIC)
        # Gentle unsharp mask
        sharpened = upscaled.filter(ImageFilter.UnsharpMask(radius=1.5, percent=120, threshold=3))
        return sharpened

    @staticmethod
    def variant_e_denoised_contrast(image: Image.Image) -> Image.Image:
        """
        Variant E: Bilateral filtering (preserves sharp edges of numbers while smoothing scan grain).
        """
        base = AdaptivePreprocessor.variant_a_normalized(image)
        cv_img = np.array(base)
        if len(cv_img.shape) == 3:
            gray = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
        else:
            gray = cv_img

        denoised = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)
        clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        return Image.fromarray(enhanced).convert("RGB")

    @classmethod
    def get_variant(cls, image: Image.Image, variant_name: str) -> Image.Image:
        variants = {
            "variant_a": cls.variant_a_normalized,
            "variant_b": cls.variant_b_contrast_clahe,
            "variant_c": cls.variant_c_adaptive_threshold,
            "variant_d": cls.variant_d_upscaled_sharpen,
            "variant_e": cls.variant_e_denoised_contrast,
        }
        fn = variants.get(variant_name.lower(), cls.variant_a_normalized)
        return fn(image)
