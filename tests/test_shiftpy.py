import pytest
import mudanca
import cv2
import numpy as np

def test_validate_greyscale():
    image = cv2.imread('examples/example.jpeg')

    with pytest.raises(Exception):
        mudanca.adjust(image, mudanca.FIT_VERTICALLY)

def test_validate_uint8():
    image = cv2.imread('examples/example.jpeg',cv2.IMREAD_GRAYSCALE )
    image = image.astype(np.float16)

    with pytest.raises(Exception):
        mudanca.adjust(image, mudanca.FIT_VERTICALLY)

def test_returns_python_error_upon_adjustment_issue():
    image = 255 * np.ones(shape=(512, 512), dtype=np.uint8)

    result =mudanca.adjust(image, mudanca.FIT_VERTICALLY)
    assert result == None