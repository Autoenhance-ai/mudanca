import cv2
import pyshift

img = cv2.imread('test.jpeg')
adjusted_img = pyshift.adjust(img)

cv2.imwrite('corrected.jpeg', adjusted_img)

cv2.imshow("Original", img)
cv2.imshow("Corrected", adjusted_img)

cv2.waitKey(0)