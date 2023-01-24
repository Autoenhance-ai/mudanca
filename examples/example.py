import cv2
import mudanca
import numpy

img = cv2.imread('example.jpeg')

lines = numpy.load(open('lines.npy', 'rb'))

height, width, _ = img.shape

lines /= 512

lines[:,0] *= width
lines[:,2] *= width
lines[:,1] *= height
lines[:,3] *= height

lines = [
    mudanca.Line(
        line[0][0],
        line[0][1],
        line[0][2],
        line[0][3],
    ) for line in lines
]

matrix, cropbox = mudanca.adjust(lines, (width, height), mudanca.FIT_VERTICALLY)

x1, y1 = cropbox[0]
x2, y2 = cropbox[1]

corrected_img = cv2.warpPerspective(img, matrix, (int(width), int(height)), flags=cv2.INTER_NEAREST)
adjusted_img = corrected_img[
    x1:x2,
    y1:y2
]

cv2.imwrite(f'corrected.jpeg', adjusted_img)