# Import libraries
import cv2
import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt
import sys

#%%

IMG_DIR = 'imgs/'

img = cv2.imread(f"{IMG_DIR}iCard0_0.jpg", cv2.IMREAD_GRAYSCALE)

kernel = np.ones((7,7), np.uint8)
img_dilate = cv2.dilate(img, kernel, iterations=1)

# Perform Canny Edge detection
img_edgedetect = cv2.Canny(img_dilate, threshold1=100, threshold2=200)

# Get Contours 
img_contours, hierarchy = cv2.findContours(img_edgedetect, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

img_with_contours = np.ones_like(img, dtype=np.uint8) * 255


max_area = 0
max_idx = -1

for idx, contour in enumerate(img_contours):
    contour_convex_hull = cv2.convexHull(contour)
    convex_hull_area = cv2.contourArea(contour_convex_hull)
    
    if(max_area < convex_hull_area):
        max_idx = idx
        max_area = convex_hull_area

if (max_idx >= 0):
    cv2.drawContours(img_with_contours, [img_contours[max_idx]], 0, (0,255,0), 2)
else:
    print("No max contour found")
    sys.exit(-1)
    
# Apply polygon approximation
approx_contour = cv2.approxPolyDP(img_contours[max_idx], 0.02 * cv2.arcLength(img_contours[max_idx], True), True)

img_poly_contour = img.copy()
cv2.drawContours(img_poly_contour, [approx_contour], -1, (0,255,0), 2)

# Get perspective transformation
W = 480
H = int(W // 1.6)
# M_perspective = cv2.getPerspectiveTransform(np.float32(approx_contour), np.float32([[0, 0], [0, H], [W, H], [W, 0]]))

hh, ww = img.shape[:2]
# perspective_img = cv2.warpPerspective(img, M_perspective, (ww,hh))[0:H,0:W]


#%%

# Show grayscale img
cv2.imshow("iCard1", img)
cv2.imshow("Dilate", img_dilate)
# Show edges
cv2.imshow('Edges', img_edgedetect)

# Show contours
cv2.imshow('Contours', img_with_contours)

# Show contours
cv2.imshow('Poly Contour', img_poly_contour)

# Perspective Transformed
# cv2.imshow('Perspectived', perspective_img)

cv2.waitKey(0)

cv2.destroyAllWindows()