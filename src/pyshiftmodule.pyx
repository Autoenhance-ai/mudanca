cimport ashift

from cpython cimport array
from libc.stdlib cimport malloc, free

import array
import cv2
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

FIT_NONE         = 0         #no Adjustments
FIT_ROTATION     = 1 << 0    # flag indicates to fit rotation angle
FIT_LENS_VERT    = 1 << 1    # flag indicates to fit vertical lens shift
FIT_LENS_HOR     = 1 << 2    # flag indicates to fit horizontal lens shift
FIT_SHEAR        = 1 << 3   # flag indicates to fit shear parameter
FIT_LINES_VERT   = 1 << 4   # use vertical lines for fitting
FIT_LINES_HOR    = 1 << 5   # use horizontal lines for fitting
FIT_LENS_BOTH    = FIT_LENS_VERT | FIT_LENS_HOR
FIT_LINES_BOTH   = FIT_LINES_VERT | FIT_LINES_HOR
FIT_VERTICALLY   = FIT_ROTATION | FIT_LENS_VERT | FIT_LINES_VERT
FIT_HORIZONTALLY = FIT_ROTATION | FIT_LENS_HOR | FIT_LINES_HOR,
FIT_BOTH         = FIT_ROTATION | FIT_LENS_VERT | FIT_LENS_HOR | FIT_LINES_VERT | FIT_LINES_HOR,
FIT_VERTICALLY_NO_ROTATION = FIT_LENS_VERT | FIT_LINES_VERT
FIT_HORIZONTALLY_NO_ROTATION = FIT_LENS_HOR | FIT_LINES_HOR
FIT_BOTH_NO_ROTATION = FIT_LENS_VERT | FIT_LENS_HOR | FIT_LINES_VERT | FIT_LINES_HOR
FIT_BOTH_SHEAR = FIT_ROTATION | FIT_LENS_VERT | FIT_LENS_HOR | FIT_SHEAR | FIT_LINES_VERT | FIT_LINES_HOR
FIT_ROTATION_VERTICAL_LINES = FIT_ROTATION | FIT_LINES_VERT
FIT_ROTATION_HORIZONTAL_LINES = FIT_ROTATION | FIT_LINES_HOR
FIT_ROTATION_BOTH_LINES = FIT_ROTATION | FIT_LINES_VERT | FIT_LINES_HOR
FIT_FLIP = FIT_LENS_VERT | FIT_LENS_HOR | FIT_LINES_VERT | FIT_LINES_HOR

def adjust(img, options, refine=cv2.LSD_REFINE_STD):

    if len(img.shape) != 2:
        raise Exception('Image must be grayscale')

    if img.dtype != np.uint8:
        raise Exception('Image must be a type of uint8')

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

    lines, widths, precision, _ = lsd.detect(img)
    height, width = img.shape

    if lines is None:
        return None

    line_count: int = lines.shape[0]

    cdef ashift.rect * rects = <ashift.rect*>malloc(sizeof(ashift.rect) * line_count)

    for line_id in range(line_count):

        x1, y1, x2, y2 = lines[line_id, 0]

        rect: ashift.rect = rects[line_id] 

        rect.x1 = x1
        rect.y1 = y1
        rect.x2 = x2
        rect.y2 = y2

        rect.width = widths[line_id]
        rect.precision = precision[line_id]

        rects[line_id] = rect

    results: float[9] = ashift.shift(
        width, height,
        line_count,
        rects,
        options
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

    free(rects)

    return matrix, cropbox