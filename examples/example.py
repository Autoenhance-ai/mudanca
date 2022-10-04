import cv2
import pyshift

img = cv2.imread('example.jpeg', 0)
matrix, cropbox = pyshift.adjust(img)

cl, cr, ct, cb = cropbox

print(matrix)
print(cropbox)

img = cv2.imread('example.jpeg')
rows, cols, ddims = img.shape

cl *= cols
cr *= cols
ct *= rows
cb *= rows

cl = int(cl)
cr = int(cr)
ct = int(ct)
cb = int(cb)

print(cl)
print(cr)
print(ct)
print(cb)

corrected_img = cv2.warpPerspective(img, matrix, (int(cols),int(rows)), flags=cv2.INTER_NEAREST)
cropped_img = corrected_img[ct:cb, cl:cr]

cv2.imwrite('corrected.jpeg', corrected_img)
cv2.imwrite('cropped.jpeg', cropped_img)

cv2.imshow("Original", img)
cv2.imshow("Corrected", corrected_img)
cv2.imshow("Cropped", cropped_img)

cv2.waitKey(0)