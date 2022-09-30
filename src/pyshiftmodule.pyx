cimport ashift
from cpython cimport array
import array
import numpy as np
from cymem.cymem cimport Pool

# TODO: 
# - Validate Input
# - Decide if we need the cymem dependency
#
def adjust(img, lsd_results: tuple):

    lines, widths, precision, _ = lsd_results
    line_count: int = lines.shape[0]
    height, width = img.shape

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

    ashift.shift(
        width, height,
        line_count,
        rects
    )

def adjust_lsd(img):
    height, width, _ = img.shape

    cdef array.array pixels = array.array('f', img.flatten())

    ashift.shift_lsd(
        pixels.data.as_floats,
        width, height,
    )