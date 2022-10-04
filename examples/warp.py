import cv2
import numpy as np

matrix = [[ 1.012507,0.022864,0.000000],
[-0.005043,1.026482,8.224804],
[-0.000000,0.000022,1.000087]]

img = cv2.imread('example.jpeg') 
rows, cols, ddims = img.shape
matrix = np.array(matrix)

corrected_img = cv2.warpPerspective(img, matrix,(int(cols),int(rows)),flags=cv2.INTER_LINEAR)

cv2.imwrite('fixed.jpeg', corrected_img)