import cv2
import pyshift
import time

img = cv2.imread('example.jpeg')
iterations = 10

print(img.dtype)

for i in range(iterations):
    adjusted_img = pyshift.adjust(img)
#     cv2.imwrite(f'corrected-{i}.jpeg', adjusted_img)