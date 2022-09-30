import cv2
import pyshift
import numpy as np

img = cv2.imread('example.jpeg', 0)
lsd = cv2.createLineSegmentDetector(0)

lsd_results = lsd.detect(img)

# print(results)
# print(results[1])
# print(results[2])

# lines = results[0]

# img_with_lines = lsd.drawSegments(img, lines)

pyshift.adjust(img, lsd_results)

# img_result = pyshift.adjust(img, lsd_results)



# print("Result")
# print(img_result)

# img_result = correction.correct(lines, img)

# M = cv2.getPerspectiveTransform(input_pts,output_pts)

# out = cv2.warpPerspective(img,M,(img.shape[1], img.shape[0]),flags=cv2.INTER_LINEAR)

# cv2.imshow("Original", img)
# cv2.imshow("LSD", img_with_lines)
# cv2.imshow("Result", out)

# cv2.waitKey(0)