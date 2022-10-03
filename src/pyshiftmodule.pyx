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

@dataclass
class Line:
    p1: np.array
    p2: np.array
    connecting_line: np.array

    length: float
    width: float
    weight: float

def point_is_in_rectangle(rectangle, point):
    
    min_x, min_y, width, height = rectangle
    max_x = min_x + width
    max_y = min_y + height

    point_x, point_y, _ = point

    return min_x <= point_x and min_y <= point_y and max_x >= point_x and max_y >= point_y


def adjust_ported(img):

    print("Shift Ported")

    height, width = img.shape

    print(f"Width: {width}")
    print(f"Height: {height}")

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

    detected_lines, widths, precs, nfa = lsd.detect(img)
    line_count, _, _ = detected_lines.shape

    print(f"Processed Line Count: {line_count}")

    lines = []

    adjustment_zone_rectangle = [
        LINE_DETECTION_MARGIN,
        LINE_DETECTION_MARGIN,
        width - LINE_DETECTION_MARGIN,
        height - LINE_DETECTION_MARGIN
    ]

    for i in range(line_count):
        line = Line()

        # store as homogeneous coordinates
        line.p1 = np.array([
            detected_lines[i][0][0],
            detected_lines[i][0][1],
            1.0
        ])

        line.p2 = np.array([
            detected_lines[i][0][2],
            detected_lines[i][0][3],
            1.0
        ])

        line.width = widths[i]

        # check for lines running along image borders and skip them.
        # these would likely be false-positives which could result
        # from any kind of processing artifacts
        if not point_is_in_rectangle(adjustment_zone_rectangle, line.p1) or not point_is_in_rectangle(adjustment_zone_rectangle, line.p2):
            continue

        # calculate homogeneous coordinates of connecting line (defined by the two points)
        line.connecting_line = np.cross(line.p1, line.p2)

        # normalaze line coordinates so that x^2 + y^2 = 1
        # (this will always succeed as L is a real line connecting two real points)
        #
        # TODO: Not sure what this does
        #
        #vec3lnorm(ashift_lines[lct].L, ashift_lines[lct].L);

        # length and width of rectangle (see LSD)
        line.length = np.linalg.norm(line.p2 - line.p1)

        # ...  and weight (= length * width * angle precision)
        line.weight = line.length * line.width * precs[i][0];

        angle = np.angle(line.p2 - line.p1)

        vertical = False
        horizontal = False

        if True: # fabsf(fabsf(angle) - 90.0f) < MAX_TANGENTIAL_DEVIATION ? 1 : 0;
            vertical = True

        if True: # fabsf(fabsf(fabsf(angle) - 90.0f) - 90.0f) < MAX_TANGENTIAL_DEVIATION ? 1 : 0;
            horizontal = True

        relevant = line.length > MIN_LINE_LENGTH

        if relevant and vertical:

            # TODO: Tag as Vertical Etc.
            #
            lines.append(line)

            #type = ASHIFT_LINE_VERTICAL_SELECTED;
            #vertical_count++;
            #vertical_weight += weight;

        elif relevant and horizontal:

            lines.append(line)

            #type = ASHIFT_LINE_HORIZONTAL_SELECTED;
            #horizontal_count++;
            #horizontal_weight += weight;

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

    results: float[8] = ashift.shift(
        width, height,
        line_count,
        rects
    )

    print(results)
    print(len(results))

    # return np.split(np.array(results), 3)
    

def adjust_lsd(img):
    height, width, _ = img.shape

    image_data = img.flatten()

    cdef array.array pixels = array.array('f', image_data)

    results: float[8] = ashift.shift_lsd(
        pixels.data.as_floats,
        width, height,
    )

    print(results)
    print(len(results))

    # return np.split(np.array(results), 3)
    