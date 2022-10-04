cimport ashift
from cpython cimport array
import array
import numpy as np
import cv2
from dataclasses import dataclass

from cymem.cymem cimport Pool

LSD_SCALE = 0.99                # LSD: scaling factor for line detection
LSD_SIGMA_SCALE = 0.6           # LSD: sigma for Gaussian filter is computed as sigma = sigma_scale/scale
LSD_QUANT = 2.0                 # LSD: bound to the quantization error on the gradient norm
LSD_ANG_TH = 22.5               # LSD: gradient angle tolerance in degrees
LSD_LOG_EPS = 0.0               # LSD: detection threshold: -log10(NFA) > log_eps
LSD_DENSITY_TH = 0.7            # LSD: minimal density of region points in rectangle
LSD_N_BINS = 1024               # LSD: number of bins in pseudo-ordering of gradient modulus
LINE_DETECTION_MARGIN = 5       # Size of the margin from the border of the image where lines will be discarded
MIN_LINE_LENGTH = 5             # the minimum length of a line in pixels to be regarded as relevant
MAX_TANGENTIAL_DEVIATION = 30   # by how many degrees a line may deviate from the +/-180 and +/-90 to be regarded as relevant

# TODO: Add Documentation And Validation
#
def adjust(img):

    lsd = cv2.createLineSegmentDetector(
        cv2.LSD_REFINE_STD,
        LSD_SCALE,
        LSD_SIGMA_SCALE,
        LSD_QUANT,
        LSD_ANG_TH,
        LSD_LOG_EPS,
        LSD_DENSITY_TH,
        LSD_N_BINS
    )

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    lines, widths, precision, _ = lsd.detect(gray)
    line_count: int = lines.shape[0]
    height, width = gray.shape

    cdef Pool mem = Pool()
    cdef ashift.rect* rects = <ashift.rect*>mem.alloc(line_count, sizeof(ashift.rect))

    for line_id in range(line_count):

        x1, y1, x2, y2 = lines[line_id, 0]

        rect: ashift.rect = rects[line_id] 

        rect.x1 = x1
        rect.y1 = y1
        rect.x2 = x2
        rect.y2 = y2

        rect.width = widths[line_id]
        rect.x = precision[line_id]

        rects[line_id] = rect

    results: float[9] = ashift.shift(
        width, height,
        line_count,
        rects
    )

    matrix = np.array(
        [
            [results[0], results[1], results[2]],
            [results[3], results[4], results[5]],
            [results[6], results[7], results[8]]
        ]
    )

    corners = np.array([
        [[0,0]],
        [[width,0]],
        [[0,height]],
        [[width,height]]
    ]).astype(np.float32)

    corners = cv2.perspectiveTransform(corners, matrix)
    points = [corners[0][0], corners[1][0], corners[2][0], corners[3][0]]

    x1 = int(max(points[0][1], points[1][1]))
    x2 = int(min(points[2][1], points[3][1]))
    y1 = int(max(points[0][0], points[2][0]))
    y2 = int(min(points[1][0], points[3][0]))

    corrected_img = cv2.warpPerspective(img, matrix, (int(width),int(height )), flags=cv2.INTER_NEAREST)

    return corrected_img[x1:x2, y1:y2]