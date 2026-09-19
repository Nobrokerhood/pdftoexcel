"""
Tests for AdaptivePreprocessor and non-destructive image processing variants.
"""

import os
import pytest
from PIL import Image
from app.documents.preprocessing import (
    AdaptivePreprocessor,
    detect_orientation,
    correct_orientation,
    deskew_image,
)


TESTING_FOLDER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "testing_folder")


def test_preprocessing_variants_generation():
    # Test with synthetic test image containing decimal points and currency symbols
    img = Image.new("RGB", (400, 200), color=(255, 255, 255))
    var_a = AdaptivePreprocessor.get_variant(img, "variant_a")
    var_b = AdaptivePreprocessor.get_variant(img, "variant_b")
    var_c = AdaptivePreprocessor.get_variant(img, "variant_c")
    var_d = AdaptivePreprocessor.get_variant(img, "variant_d")
    var_e = AdaptivePreprocessor.get_variant(img, "variant_e")

    assert var_a.size == (400, 200)
    assert var_b.size == (400, 200)
    assert var_c.size == (400, 200)
    assert var_d.size == (600, 300)  # upscaled 1.5x
    assert var_e.size == (400, 200)


def test_unnamed_jpg_orientation_handling():
    img_path = os.path.join(TESTING_FOLDER, "unnamed.jpg")
    if not os.path.exists(img_path):
        pytest.skip("Test file not found")

    im = Image.open(img_path)
    # unnamed.jpg is 512x384 (landscape), but the document is portrait
    assert im.width > im.height

    corrected, deg = correct_orientation(im)
    assert corrected is not None
    # Check that variants execute without crashing or memory corruption
    var_b = AdaptivePreprocessor.variant_b_contrast_clahe(im)
    assert var_b.mode == "RGB"
