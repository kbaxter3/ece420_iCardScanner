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
TRUE_UIN = "659750250" 

if not video.isOpened():
    print('Cannot open video')
    sys.exit()
    
# TESTING: variables to store accuracy measurement
user_input_accurate_frames = []
accurate_uin = []

    
# for filename in jpg_names:
#     img = cv2.imread(f"{IMG_DIR}{filename}", cv2.IMREAD_GRAYSCALE)

while True:
    read_success, img = video.read()
    if not read_success:
        print('End of video')
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # TESTING: save accuracy mesurements to csv file
        user_input_accurate_frames = user_input_accurate_frames[1:]
        accurate_uin = accurate_uin[1:]
        print(f"Card Detection Accuracy = {sum(user_input_accurate_frames) / len(user_input_accurate_frames)}")
        print(f"UIN Detection Accuracy = {sum(accurate_uin) / sum(user_input_accurate_frames)}")
        
        
        sys.exit()
        
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    
    img_poly_contour = img.copy()
    approx_contour = get_bounding_quadrangle(img)
    
    if(type(approx_contour) == np.ndarray and approx_contour.shape[0] == 4):
        cv2.drawContours(img_poly_contour, [approx_contour], -1, (0,255,0), 2)
        
        # Get top left corner based on side lengths
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
        PAD = 10
        M_perspective = cv2.getPerspectiveTransform(np.float32(approx_contour), np.float32([[PAD, PAD], [PAD, H+PAD], [W+PAD, H+PAD], [W+PAD, PAD]]))
        
        hh, ww = img.shape[:2]
        perspective_img = cv2.warpPerspective(img, M_perspective, (ww,hh))[0:H+2*PAD,0:W+2*PAD]
        cv2.imshow('Perspectived', perspective_img)
        cv2.imshow("Annotated", img_poly_contour)
        
        # TESTING: user input for card detection accuracy
        user_input = input("Enter 1 for correct identification of a card and 0 if not: ")
        user_in = False
        while(user_in == False):
            if user_input.isdigit():
                user_input_accurate_frames.append(int(user_input))
                user_in = True
            else:
                user_input = input("Invalid input, please enter a digit between 0-1: ")
        
        # TESTING: card detection accuracy by evaluating two of the angles of the contour
        # ANGLE_THESHOLD = 30
        # vec1 = approx_contour[0,0] - approx_contour[1,0]
        # vec2 = approx_contour[0,0] - approx_contour[-1,0]
        # vec3 = approx_contour[2,0] - approx_contour[1,0]
        # vec4 = approx_contour[2,0] - approx_contour[3,0]
        # dot_product = np.dot(vec1, vec2)
        # magnitude_A = np.linalg.norm(vec1)
        # magnitude_B = np.linalg.norm(vec2)
        # angle_1 = np.abs(np.degrees(np.arccos(dot_product / (magnitude_A * magnitude_B))))
        # dot_product = np.dot(vec3, vec4)
        # magnitude_A = np.linalg.norm(vec3)
        # magnitude_B = np.linalg.norm(vec4)
        # angle_2 = np.abs(np.degrees(np.arccos(dot_product / (magnitude_A * magnitude_B))))
        # if((angle_1 + angle_2) > 180 - ANGLE_THESHOLD and (angle_1 + angle_2) < 180 + ANGLE_THESHOLD):
        #     user_input_accurate_frames.append(1)
        #     print("Good frame")
        # else:
        #     # User input to check for mistakes
        #     user_input = input("Bad frame, enter 1 to switch to good frame: ")
        #     if(user_input == 1):
        #         user_input_accurate_frames.append(1)
        #     else:
        #         user_input_accurate_frames.append(0)
        #     print(f"Bad frame: {angle_1}, {angle_2}")
        #     cv2.waitKey(0)
            
        
        
        user_input_accurate_frames.append(1) # delete later
            
            
        # Tesseract text recognition
        # Crop the card to the UIN text
        uin_x_start = (int)(W*2/11)+PAD
        uin_width = (int)(W/4)
        uin_y_start = H - (int)(H/10)+PAD
        uin_height = (int)(H/10)
        img_uin = perspective_img[uin_y_start : uin_y_start + uin_height, uin_x_start : uin_x_start + uin_width]
        cv2.imshow("NetID Cropped", img_uin)
        uin_text = extract_text_from_image(img_uin)
        uin_text = uin_text.replace("\n","")
        
        # If the text identified is not the uin, try flipping the image
        if(not uin_text.isdigit() and len(uin_text) != 9):
            print(f"UIN: {uin_text}, length: {len(uin_text)}")
            perspective_img = perspective_img[::-1, ::-1]
            img_uin = perspective_img[uin_y_start : uin_y_start + uin_height, uin_x_start : uin_x_start + uin_width]
            uin_text = extract_text_from_image(img_uin)
            uin_text = uin_text.replace("\n","")
            cv2.imshow("NetID Cropped", img_uin)
            cv2.imshow('Perspectived', perspective_img)
            print("rotated")
            
            
        # If the text still does not resemble a uin, print no text detected
        if(not uin_text.isdigit() and len(uin_text) != 9):
            print("No UIN Detected")
        
        print(f"UIN: {uin_text}, length: {len(uin_text)}")
        
        # TESTING record if UIN is correct
        if(uin_text == TRUE_UIN):
            print("Correct UIN")
            accurate_uin.append(1)
        else:
            accurate_uin.append(0)
        
    else:
        print("No card detected")
        user_input_accurate_frames.append(0)
        
        
        
    cv2.waitKey(2)

#%%
cv2.waitKey(0)

cv2.destroyAllWindows()