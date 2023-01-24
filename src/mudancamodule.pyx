cimport ashift

from cpython cimport array
from libc.stdlib cimport malloc, free

from dataclasses import dataclass
import array
import cv2
import numpy as np

FIT_NONE         = 0        #no Adjustments
FIT_ROTATION     = 1 << 0   # flag indicates to fit rotation angle
FIT_LENS_VERT    = 1 << 1   # flag indicates to fit vertical lens shift
FIT_LENS_HOR     = 1 << 2   # flag indicates to fit horizontal lens shift
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

@dataclass
class Line:
    x1: float
    y1: float

    x2: float
    y2: float

    width: float = 1.0
    precision: float = 1.0

def adjust(lines, size, options):

    width, height = size

    if lines is None:
        return None

    line_count: int = len(lines)

    cdef ashift.rect * rects = <ashift.rect*>malloc(sizeof(ashift.rect) * line_count)

    for line_id, line in enumerate(lines):

        rect: ashift.rect = rects[line_id] 

        rect.x1 = line.x1
        rect.y1 = line.y1
        rect.x2 = line.x2
        rect.y2 = line.y2

        rect.width = line.width
        rect.precision = line.precision

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