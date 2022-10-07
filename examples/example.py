import cv2
import shiftpy

img = cv2.imread('example.jpeg')
iterations = 10

for i in range(iterations):

    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    matrix, cropbox = shiftpy.adjust(grey, shiftpy.FIT_VERTICALLY)
    
    height, width, _ = img.shape
    x1, y1 = cropbox[0]
    x2, y2 = cropbox[1]

    corrected_img = cv2.warpPerspective(img, matrix, (int(width), int(height)), flags=cv2.INTER_NEAREST)
    adjusted_img = corrected_img[
        x1:x2,
        y1:y2
    ]

    cv2.imwrite(f'corrected-{i}.jpeg', adjusted_img)