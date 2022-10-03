import cv2
import pyshift
import numpy as np

img = cv2.imread('example.jpeg', 0)

# These Parameters are taken from Darktable's LSD
#
lsd = cv2.createLineSegmentDetector(
    1,
    0.99,
    0.6,
    2.0,
    22.5,
    0.0,
    0.7,
    1024
)

lsd_results = lsd.detect(img)
pyshift.adjust(img, lsd_results)

img = cv2.imread('example.jpeg') 
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
pyshift.adjust_lsd(img)


# M = cv2.getPerspectiveTransform(input_pts,output_pts)
# out = cv2.warpPerspective(img,M,(img.shape[1], img.shape[0]),flags=cv2.INTER_LINEAR)

# cv2.imshow("Original", img)
# cv2.imshow("LSD A", img_with_lines)
# cv2.imshow("LSD B", img_with_linesB)
# cv2.imshow("Result", out)

# cv2.waitKey(0)