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

matrix = pyshift.adjust_ported(img)
rows, cols = img.shape
corrected_img_c = cv2.warpPerspective(img, matrix,(int(cols),int(rows)),flags=cv2.INTER_LINEAR)

lsd_results = lsd.detect(img)
matrix = pyshift.adjust(img, lsd_results)

rows, cols = img.shape

corrected_img_a = cv2.warpPerspective(img, matrix,(int(cols),int(rows)),flags=cv2.INTER_LINEAR)


img = cv2.imread('example.jpeg') 
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
matrix = pyshift.adjust_lsd(img)

corrected_img_b = cv2.warpPerspective(img, matrix,(int(cols),int(rows)),flags=cv2.INTER_LINEAR)


cv2.imshow("Original", img)
cv2.imshow("LSD A", corrected_img_a)
cv2.imshow("LSD B", corrected_img_b)
cv2.imshow("LSD C", corrected_img_c)

cv2.waitKey(0)