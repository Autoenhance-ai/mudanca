import cv2
import pyshift

img = cv2.imread('example.jpeg', 0)
matrix = pyshift.adjust(img)
print(matrix)

img = cv2.imread('example.jpeg')
rows, cols, ddims = img.shape


corrected_img = cv2.warpPerspective(img, matrix, (int(cols),int(rows)))

cv2.imwrite('fixed.jpeg', corrected_img)
cv2.imshow("Original", img)
cv2.imshow("LSD", corrected_img)

cv2.waitKey(0)