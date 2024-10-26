# Import libraries
import cv2
import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt
import sys
import os
import pytesseract

#%%

def get_bounding_quadrangle(img):
    kernel = np.ones((3,3), np.uint8)
    img_dilate = cv2.dilate(img, kernel, iterations=1)
    # cv2.imshow("Dilate", img_dilate)
    
    # Perform Canny Edge detection
    img_edgedetect = cv2.Canny(img_dilate, threshold1=100, threshold2=200)
    # cv2.imshow('Edges', img_edgedetect)
    
    # Get Contours 
    img_contours, hierarchy = cv2.findContours(img_edgedetect, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    img_with_contours = np.ones_like(img, dtype=np.uint8) * 255
    
    max_area = 0
    max_idx = -1
    
    for idx, contour in enumerate(img_contours):
        contour_convex_hull = cv2.convexHull(contour)
        convex_hull_area = cv2.contourArea(contour_convex_hull)
        # print(contour_convex_hull)
        
        if(max_area < convex_hull_area):
            max_idx = idx
            max_area = convex_hull_area
    
    if (max_idx >= 0):
        cv2.drawContours(img_with_contours, [img_contours[max_idx]], 0, (0,255,0), 2)
        # cv2.imshow('Contours no hull', img_with_contours)
        cv2.drawContours(img_with_contours, [cv2.convexHull(img_contours[max_idx])], 0, (0,255,0), 2)
        # cv2.imshow('Contours', img_with_contours)
    else:
        print("No max contour found")
        return -1
        
    # Apply polygon approximation
    for err in np.linspace(0.005, 0.09, 80):
        approx_contour = cv2.approxPolyDP(cv2.convexHull(img_contours[max_idx]), err * cv2.arcLength(img_contours[max_idx], True), True)
        # print(f"err: {err}, numpt: {approx_contour.shape}")
        if(approx_contour.shape[0] == 4):
            break
    
    return approx_contour

# From ChatGPT
def list_jpg_files(directory):
    try:
        # List to hold .jpg filenames
        jpg_files = []

        # Loop through the files in the directory
        for filename in os.listdir(directory):
            # Check if the file has a .jpg extension
            if filename.lower().endswith('.jpg'):
                jpg_files.append(filename)

        return jpg_files

    except FileNotFoundError:
        print("The specified directory does not exist.")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

def extract_text_from_image(image):
    # Convert the image to RGB (Tesseract requires RGB images)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Use Tesseract to extract text
    text = pytesseract.image_to_string(rgb_image)

    return text


#%%

IMG_DIR = 'imgs/'

jpg_names = list_jpg_files(IMG_DIR)

video = cv2.VideoCapture(f'{IMG_DIR}icard0_vid.mp4')

if not video.isOpened():
    print('Cannot open video')
    sys.exit()
    
    
# for filename in jpg_names:
#     img = cv2.imread(f"{IMG_DIR}{filename}", cv2.IMREAD_GRAYSCALE)

while True:
    read_success, img = video.read()
    if not read_success:
        print('End of video')
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        sys.exit()
        
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    
    img_poly_contour = img.copy()
    approx_contour = get_bounding_quadrangle(img)
    
    if(type(approx_contour) == np.ndarray and approx_contour.shape[0] == 4):
        cv2.drawContours(img_poly_contour, [approx_contour], -1, (0,255,0), 2)
        
        # Gettop left corner based on side lengths
        vec1 = approx_contour[0,0] - approx_contour[1,0]
        vec2 = approx_contour[0,0] - approx_contour[-1,0]
        
        # Get CW/CCW by cross product magnitude
        if(vec1[0]*vec2[1] - vec1[1]*vec2[0] > 0):
            approx_contour = approx_contour[::-1]
            
        vec1 = approx_contour[0,0] - approx_contour[1,0]
        vec2 = approx_contour[0,0] - approx_contour[-1,0]
        if(np.linalg.norm(vec1) > np.linalg.norm(vec2)):
            approx_contour = np.roll(approx_contour, 1, axis=0)
        
        # Get perspective transformation
        W = 480
        H = int(W // 1.6)
        M_perspective = cv2.getPerspectiveTransform(np.float32(approx_contour), np.float32([[0, 0], [0, H], [W, H], [W, 0]]))
        
        hh, ww = img.shape[:2]
        perspective_img = cv2.warpPerspective(img, M_perspective, (ww,hh))[0:H,0:W]
        cv2.imshow('Perspectived', perspective_img)
        cv2.imshow("Annotated", img_poly_contour)
        
        
        print(extract_text_from_image(perspective_img))
    else:
        print("No card detected")
        
    cv2.waitKey(2)

#%%
cv2.waitKey(0)

cv2.destroyAllWindows()