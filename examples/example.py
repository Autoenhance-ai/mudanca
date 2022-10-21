import cv2
import mudanca

img = cv2.imread('example.jpeg')
iterations = 10

for i in range(iterations):

    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    lsd = cv2.createLineSegmentDetector(
        cv2.LSD_REFINE_ST,
        LSD_SCALE,
        LSD_SIGMA_SCALE, 
        LSD_QUANT,
        LSD_ANG_TH,
        LSD_LOG_EPS,
        LSD_DENSITY_TH,
        LSD_N_BINS
    )

    lines, _, _, _ = lsd.detect(img)

    height, width, _ = img.shape
    matrix, cropbox = mudanca.adjust(lines, (width, height), mudanca.FIT_VERTICALLY)
    
    x1, y1 = cropbox[0]
    x2, y2 = cropbox[1]

    corrected_img = cv2.warpPerspective(img, matrix, (int(width), int(height)), flags=cv2.INTER_NEAREST)
    adjusted_img = corrected_img[
        x1:x2,
        y1:y2
    ]

    cv2.imwrite(f'corrected-{i}.jpeg', adjusted_img)