# Import libraries
import cv2
import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt
import sys
import os
import pytesseract

#%%
def dist_point_line(x1, y1, x2, y2, x0, y0):
    return ((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1) / np.sqrt((y2-y1)**2 + (x2-x1)**2)

def quadrangle_approx(orig_contour):
    # Uses L1 (manhanttan distance) instead of L2 norm (euclidean) since only need ranking, not actual distances
    distances = np.zeros(len(orig_contour))
    for i in range(len(orig_contour)):
        distances[i] = np.abs(orig_contour[i][0][0] - orig_contour[i-1][0][0]) + np.abs(orig_contour[i][0][1] - orig_contour[i-1][0][1])

    # 4 lines, each one is (x1,y1,x2,y2)
    lines = np.zeros((4,4), dtype=np.int32)

    corners = np.zeros((4,2), dtype=np.int32)
    slope_array = np.zeros(4)
    y_int_array = np.zeros(4)

    # Find 4 longest, non-colinear lines in contours (will segments on card edges)
    COLINEAR_CUTOFF = 10
    next_line = 0
    for i in range(len(distances)):
        max_idx = np.argsort(distances)[-(i+1)]

        # Save the line endpoints
        lines[next_line][0:2] = orig_contour[max_idx][0]
        lines[next_line][2:4] = orig_contour[max_idx-1][0]

        slope_denom = (lines[next_line][2]-lines[next_line][0])
        if (slope_denom == 0):
            slope = (lines[next_line][3]-lines[next_line][1]) / 0.00001
        else:
            slope = (lines[next_line][3]-lines[next_line][1]) / slope_denom

        y_int = lines[next_line][1] - slope * lines[next_line][0]

        # Check if colinear with a line already in our set
        colinear = False
        for j in range(next_line):
            if (dist_point_line(*(lines[j]), lines[next_line][0], lines[next_line][1]) < COLINEAR_CUTOFF and 
                dist_point_line(*(lines[j]), lines[next_line][2], lines[next_line][3]) < COLINEAR_CUTOFF):
                colinear = True
                break


        if(not colinear):
            slope_array[next_line] = slope
            y_int_array[next_line] = y_int

            next_line += 1
            if(next_line == 4):
                break

    # Find opposite opposite to lines[0]
    if(np.abs(slope_array[0]) < 0.001):
        slope_array[0] = 0.001

    opposite_idx = 1
    slope_ratio = np.abs(np.abs(slope_array[1] / slope_array[0]) - 1)

    for i in range(2,4):
        new_slope_ratio = np.abs(np.abs(slope_array[i] / slope_array[0]) - 1)
        if(new_slope_ratio < slope_ratio):
            slope_ratio = new_slope_ratio
            opposite_idx = i

    # Find 2 line indicies perpendicular to lines[0]
    if(opposite_idx == 1):
        adj_idx = np.array((2,3))
    elif(opposite_idx == 2):
        adj_idx = np.array((1,3))
    else:
        adj_idx = np.array((1,2))

    # Calculate corners
    for i in range(2):
        # avoid rounding errors with x_tmp
        slope_denom = (slope_array[0] - slope_array[adj_idx[i]])
        if(slope_denom == 0):
            slope_denom = 0.001
        x_tmp = (y_int_array[adj_idx[i]] - y_int_array[0]) / slope_denom
        corners[i][0] = x_tmp
        corners[i][1] = slope_array[0] * x_tmp + y_int_array[0]

        slope_denom = (slope_array[opposite_idx] - slope_array[adj_idx[i]])
        if(slope_denom == 0):
            slope_denom = 0.001
        x_tmp = (y_int_array[adj_idx[i]] - y_int_array[opposite_idx]) / slope_denom
        corners[3-i][0] = x_tmp
        corners[3-i][1] = slope_array[opposite_idx] * x_tmp + y_int_array[opposite_idx]
    
    corner_countor = np.zeros((4,1,2), dtype=np.int32)
    
    for i in range(4):
        corner_countor[i][0] = corners[i]
    
    return corner_countor

# Function that returns the contour of the largest rectangular object in the frame
def get_bounding_quadrangle(img):
    kernel = np.ones((3,3), np.uint8)
    img_dilate = cv2.dilate(img, kernel, iterations=1)
    # cv2.imshow("Dilate", img_dilate)
    
    # Perform Canny Edge detection
    img_edgedetect = cv2.Canny(img_dilate, threshold1=100, threshold2=200)
    cv2.imshow('Edges', img_edgedetect)
    
    # Get Contours 
    img_contours, hierarchy = cv2.findContours(img_edgedetect, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    img_with_contours = np.ones_like(img, dtype=np.uint8) * 255
    
    # Find the contour with the maximum area
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
        cv2.imshow('Contours', img_with_contours)
    else:
        print("No max contour found")
        return -1

    approx_contour = quadrangle_approx(cv2.convexHull(img_contours[max_idx]))
    
    return approx_contour

# Uses tesseract OCR to return all the characters in an image, thresholding applied as pre processing for tesseract
def extract_text_from_image(image):
    # Threhold the image
    blur = cv2.GaussianBlur(image,(3,3),0)
    ret3,img_uin = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # Convert the image to RGB (Tesseract requires RGB images)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Use Tesseract to extract text, signle line mode (psm7)
    text = pytesseract.image_to_string(rgb_image, config='--psm 7')
    
    cv2.imshow("thresh_crop", img_uin)

    return text


#%%

IMG_DIR = 'imgs/'

# Video for testing: Change the video to test and the accurate UIN
video = cv2.VideoCapture(f'{IMG_DIR}icard1_moving.mp4')
TRUE_UIN = "653750367" 

if not video.isOpened():
    print('Cannot open video')
    sys.exit()

# Set up output video file writer (records annotated frames to an mp4 file, for presentation)
# output_video_path = f'{IMG_DIR}icard0_outputvid_1.mp4'
# frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fps = 2 #video.get(cv2.CAP_PROP_FPS)
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 file
# out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    
# TESTING: variables to store accuracy measurement in each frame
user_input_accurate_frames = [] # We manually indicate 1 for an accurately detected card and 0 if not
accurate_uin = [] # Records if text detected in the frame matches the true UIN (1) or not (0)
false_uin = [] # Records if the text detected contains digits of the expected length of a UIN, but is incorrect (1)

    
# for filename in jpg_names:
#     img = cv2.imread(f"{IMG_DIR}{filename}", cv2.IMREAD_GRAYSCALE)

# Loop through each frame of the video
frame_num = 0
while True:
    read_success, img_color = video.read()
    if not read_success:
        # We have reached the last frame of the video
        print('End of video')
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # For the output annotated video
        #out.release()
        
        # TESTING: accuracy metrics
        # print(f"Card Detection Accuracy = {sum(user_input_accurate_frames) / len(user_input_accurate_frames)}")
        # print(f"UIN Detection Accuracy = {sum(accurate_uin) / sum(user_input_accurate_frames)}")
        # print(f"False UIN Detection = {sum(false_uin) / len(user_input_accurate_frames)}")
        # print(f"Num frames = {len(user_input_accurate_frames)}")
        
        sys.exit()
    # skip the first frame of the video
    frame_num += 1
    if(frame_num == 1):
        continue
    
    
    # Switch to grayscale for processing
    img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    
    img_poly_contour = img_color
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
        M_perspective = cv2.getPerspectiveTransform(np.float32(approx_contour), np.float32([[0, 0], [0, H], [W, H], [W, 0]]))
        
        hh, ww = img.shape[:2]
        perspective_img = cv2.warpPerspective(img, M_perspective, (ww,hh))[0:H,0:W]
        perspective_img_color = cv2.warpPerspective(img_color, M_perspective, (ww,hh))[0:H,0:W]
        
        # Orient the card putting the orange side to the left
        cv2.line(perspective_img_color, (W//8,0), (W//8, H), (0,0,255), 2)
        cv2.imshow("Red Channel", perspective_img_color[0:H, 0:W, 2])
        red_left = np.sum(perspective_img_color[:, 0:W//8, 2].flatten())
        red_right = np.sum(perspective_img_color[:, W-W//8:W, 2].flatten())
        print(f"Red Channel: Ratio = {red_left/red_right}, Left Side= {red_left}, Right Side = {red_right}")
        if(red_left/red_right < 1):
            perspective_img = perspective_img[::-1, ::-1]
        
        # Tesseract text recognition
        # Crop the card to the UIN text
        uin_x_start = (int)(W*20/100)
        uin_width = (int)(W/4)
        uin_y_start = H - (int)(H/10)
        uin_height = (int)(H/10)
        img_uin = perspective_img[uin_y_start : uin_y_start + uin_height, uin_x_start : uin_x_start + uin_width]
        uin_text = extract_text_from_image(img_uin)
        uin_text = uin_text.replace("\n","")
        cv2.imshow("NetID Cropped", img_uin)
        
            
        # If the text still does not resemble a uin, print no text detected
        if(not uin_text.isdigit() or len(uin_text) != 9):
            print("No UIN Detected")
        
        print(f"UIN: {uin_text}, length: {len(uin_text)}")
        
        cv2.putText(img_poly_contour, f"UIN: {uin_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        
        cv2.imshow('Perspectived', perspective_img)
        cv2.imshow("Annotated", img_poly_contour)
        
        #out.write(img_poly_contour)
        
        # TESTING: user input for card detection accuracy
        # user_input = input("Enter 1 for correct identification of a card and 0 if not: ")
        # user_in = False
        # while(user_in == False):
        #     if user_input.isdigit():
        #         user_input_accurate_frames.append(int(user_input))
        #         user_in = True
        #     else:
        #         user_input = input("Invalid input, please enter a digit between 0-1: ")
        
        # TESTING record if UIN is correct
        if(uin_text == TRUE_UIN):
            print("Correct UIN")
            accurate_uin.append(1)
            false_uin.append(0)
        else: 
            accurate_uin.append(0)
            if(len(uin_text) == len(TRUE_UIN) and uin_text.isdigit()):
                false_uin.append(1)
            else:
                false_uin.append(0)
        
        
    else:
        print("No card detected")
        
        
    # cv2.waitKey(0)        
    cv2.waitKey(1)

#%%
cv2.waitKey(0)

cv2.destroyAllWindows()