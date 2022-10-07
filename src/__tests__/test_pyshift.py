import pytest
import pyshift
import cv2
import numpy as np

def test_validate_greyscale():
    image = cv2.imread('examples/example.jpeg')

    with pytest.raises(Exception):
        pyshift.adjust(image)

def test_validate_uint8():
    image = cv2.imread('examples/example.jpeg',cv2.IMREAD_GRAYSCALE )
    image = image.astype(np.float16)

    with pytest.raises(Exception):
        pyshift.adjust(image)

def test_returns_python_error_upon_adjustment_issue():
    image = 255 * np.ones(shape=(512, 512), dtype=np.uint8)

    result =pyshift.adjust(image)
    assert result == None