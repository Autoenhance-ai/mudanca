cimport ashift

from cpython cimport array
from libc.stdlib cimport malloc, free

import array
import cv2
import logging
import numpy as np

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

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# TODO: Add Documentation + Validation
#
def adjust(img, refine=cv2.LSD_REFINE_STD):

    lsd = cv2.createLineSegmentDetector(
        refine,
        LSD_SCALE,
        LSD_SIGMA_SCALE, 
        LSD_QUANT,
        LSD_ANG_TH,
        LSD_LOG_EPS,
        LSD_DENSITY_TH,
        LSD_N_BINS
    )

    logger.info("Sharpening Image")

    smoothed = cv2.GaussianBlur(img, (9, 9), 10)
    unsharp = cv2.addWeighted(img, 1.5, smoothed, -0.5, 0)

    logger.info("Detecting Lines")

    lines, widths, precision, _ = lsd.detect(img)
    line_count: int = lines.shape[0]
    height, width = img.shape

    logger.info("Collecting Lines")

    cdef ashift.rect * rects = <ashift.rect*>malloc(sizeof(ashift.rect) * line_count)

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

    logger.info("Calcculating Adjustment")

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

    src_points = np.array([
        [[0,0]],
        [[width,0]],
        [[0,height]],
        [[width,height]]
    ]).astype(np.float32)

    dst_points = cv2.perspectiveTransform(src_points, matrix)

    x1 = int(max(dst_points[0][0][1], dst_points[1][0][1]))
    x2 = int(min(dst_points[2][0][1], dst_points[3][0][1]))
    y1 = int(max(dst_points[0][0][0], dst_points[2][0][0]))
    y2 = int(min(dst_points[1][0][0], dst_points[3][0][0]))

    cropbox = [
        (x1, y1),
        (x2, y2)
    ]

    logger.info("Cleaning Up")
    free(rects)

    return matrix, cropbox