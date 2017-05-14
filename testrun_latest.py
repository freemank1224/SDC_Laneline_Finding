# -*- coding: utf-8 -*-
"""
CarND Project4 Advanced Lane Line Finding

Created on Thu Mar 23 11:14:27 2017

@author: Dyson Freeman
"""
'''
The goals / steps of this project are the following:

    1.Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
    2.Apply a distortion correction to raw images.
    3.Use color transforms, gradients, etc., to create a thresholded binary image.
    4.Apply a perspective transform to rectify binary image ("birds-eye view").
    5.Detect lane pixels and fit to find the lane boundary.
    6.Determine the curvature of the lane and vehicle position with respect to center.
    7.Warp the detected lane boundaries back onto the original image.
    8.Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
'''
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import sys


global counter
counter = 0


'''
STEP1: Camera Calibration
       One just need to call this function to calibrate the camera once. Then the
       function will store calibration parameters to a pickle file.
       ONE extra parameters should be modified for other users:
           * The output pickle file name and relative path
'''

def cameraCalibration(chass_img_list, numH, numV):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((numH*numV,3), np.float32)
    objp[:,:2] = np.mgrid[0:numH, 0:numV].T.reshape(-1,2)
    
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.
 
    # Step through the list and search for chessboard corners
    for fname in chass_img_list:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (numH,numV),None)
    
        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
    
            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (numH,numV), corners, ret)
            cv2.imshow('img',img)
            cv2.waitKey(100)
    
    cv2.destroyAllWindows()

    # Get the camera characteristics by doing calibration and store parameters for later use
    img_size = (img.shape[1], img.shape[0])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    pickle.dump(dist_pickle, open( "./cameraData/calibrations.p", "wb"))

'''
STEP2: Image Undistortion 
       Only correct the distortion introduced by the camera lens, perspective distort
       is not included.
'''
def undistortion(img, mtx, dist):
    # Test undistortion on an image
    
    dst = cv2.undistort(img, mtx, dist, None, mtx)
#    cv2.imwrite('./camera_cal/test_undist.jpg',dst)

    return dst
    
#image_name_list = glob.glob('./test_images/test*.jpg')
#
#for idx, image in enumerate(image_name_list):
#    image = plt.imread(image)
#    #print(idx, image)
#    undistort = undistortion(image, mtx, dist)
#    outfilename = 'undist_test' + str(idx) + '.jpg'
#    print(outfilename)
#    cv2.imwrite('./output_images/' + outfilename, undistort)
    
    # Visualize undistortion Obly for testing!
#    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
#    ax1.imshow(image)
#    ax1.set_title('Original Image', fontsize=30)
#    ax2.imshow(undistort)
#    ax2.set_title('Undistorted Image', fontsize=30)

'''
Step3. Thresholding Images
'''
# This thresholdingBinary() function takes sobel matrix, the magnitude of gredients,
# the orientation of gradients and the HLS color transformation into account, to derive a 
# thresholding binary image in which the lane lines can be clearly seen.

def thresholdingBinary(img, s_thresh=(170, 250), sx_thresh=(100, 255), sy_thresh=(100,255), dir_thresh=(0.6,0.7), mag_thresh=(30,200)):
    img = np.copy(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Convert to HSV color space and separate the V channel
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    h_channel = hsv[:,:,0]
    l_channel = hsv[:,:,1]
    s_channel = hsv[:,:,2]
#    plt.imshow(s_channel)
    
    # Sobel x
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobelx = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Sobel y
    sobely = cv2.Sobel(gray, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobely = np.absolute(sobely) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobely = np.uint8(255*abs_sobely/np.max(abs_sobely))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobelx)
    sxbinary[(scaled_sobelx >= sx_thresh[0]) & (scaled_sobelx <= sx_thresh[1])] = 1

    # Threshold y gradient
    sybinary = np.zeros_like(scaled_sobely)
    sxbinary[(scaled_sobely >= sy_thresh[0]) & (scaled_sobely <= sy_thresh[1])] = 1
             
    # Threshold direction gradient
    direct = np.arctan2(abs_sobely, abs_sobelx)
    
    # Threshold direction
    dir_binary = np.zeros_like(direct)
    dir_binary[(direct > np.min(dir_thresh)) & (direct < np.max(dir_thresh))] = 1
               
    # Magnitude gradient
    mag = np.sqrt(np.square(sobelx) + np.square(sobely))
    scaled_mag = np.uint8(255 * mag/np.max(mag))
    mag_binary = np.zeros_like(mag)
    mag_binary[(scaled_mag > np.min(mag_thresh)) & (scaled_mag < np.max(mag_thresh))] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
#    h_binary = np.zeros_like(h_channel)
#    h_binary[(h_channel >= s_thresh[0]) & (h_channel <= s_thresh[1])] = 1
    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    color_binary = np.dstack((mag_binary, sxbinary, s_binary))
    mono_binary = np.zeros_like(gray)
    mono_binary[(s_binary == 1) | (sybinary == 1) | (sxbinary == 1) | (mag_binary == 1) | (dir_binary == 1)] = 1
#    print(color_binary.shape)
#    return color_binary
    return mono_binary

# Using this function to get a thresholding image
#image_name_list = glob.glob('./output_images/undist_test*.jpg')
#for idx, image in enumerate(image_name_list):
#    image = cv2.imread(image)
#    binary_image = thresholdingBinary(image)
#    outfilename = 'binary_test' + str(idx) + '.jpg'
#    print(outfilename)
#    cv2.imwrite('./output_images/' + outfilename, binary_image)
##    warped_img = perspectiveTrans(binary_image)    
#    
#    # Plot the result
#    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
#    f.tight_layout()
#    
#    ax1.imshow(image)
#    ax1.set_title('Original Image', fontsize=40)
#    
#    ax2.imshow(binary_image)
#    ax2.set_title('Pipeline Result', fontsize=40)
#    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

'''
Step4. Perspective Transform
'''
# This calibration only take two straight line images
def perspectiveTrans(img):
    img_size = (img.shape[1], img.shape[0])
    
    src = np.float32([[275.93, 664.08],
                      [1034.93, 664.08],
                      [708.17,  462.10],
                      [576.17, 462.10]])
    
    dst = np.float32([[275.93, 664.08],
                      [1034.93, 664.08],
                      [1034.93, 64.08],
                      [275.93, 64.08]])
    
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    perspect_pickle = {}
    perspect_pickle['M'] = M
    perspect_pickle['Minv'] = Minv
    pickle.dump(perspect_pickle, open( "./cameraData/perspective.p", "wb"))
    
    return warped

'''
Step5. Sliding Window to Find lanes
'''
def slidingWindows(binary_warped, nwindows=9, margin=60, minpix=100):
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    # Choose the number of sliding windows
#    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
#    print(nonzeroy.shape())
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
#    margin = 100
    # Set minimum number of pixels found to recenter window
#    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - int(0.8 * margin)
        win_xleft_high = leftx_current + int(1.2 * margin)
        win_xright_low = rightx_current - int(0.8 * margin)
        win_xright_high = rightx_current + int(1.2 * margin)
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        
#        print(win_xleft_low, win_xleft_high, win_y_low, win_y_high)
#        print((((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high))).shape)
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        
#        print(left_lane_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
    
    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    
#    print(left_lane_inds.shape)
    
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    # Fit a second order polynomial to each
    left_fit, left_fit_res, _, _, _ = np.polyfit(lefty, leftx, 2, full=True)
    right_fit, right_fit_res, _, _, _ = np.polyfit(righty, rightx, 2, full=True)
#    print(left_fit)
#    print(right_fit)
    # Fit a polynomial using the actual length of road
    ym_per_px = 30/720
    xm_per_px = 3.7/700
    
    left_fit_meter, left_fit_meter_res, _, _, _ = np.polyfit(lefty*ym_per_px, leftx*xm_per_px, 2, full=True)
    right_fit_meter, right_fit_meter_res, _, _, _ = np.polyfit(righty*ym_per_px, rightx*xm_per_px, 2, full=True)    
    
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    resultDict = {'left_fit': left_fit, 'right_fit':right_fit,
    'left_fit_meter':left_fit_meter, 'right_fit_meter':right_fit_meter,
    'left_fit_res':left_fit_res, 'right_fit_res':right_fit_res,
    'left_fit_meter_res':left_fit_meter_res, 'right_fit_meter_res':right_fit_meter_res}
    
    return resultDict, out_img


def lineFinding(binary_warped, left_fit, right_fit, margin=55): 
    # It's now much easier to find line pixels!
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)
#    out_img = binary_warped
#    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  
    
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit_2, left_fit_res_2, _, _, _ = np.polyfit(lefty, leftx, 2, full=True)
    right_fit_2, right_fit_res_2, _, _, _ = np.polyfit(righty, rightx, 2, full=True)
    
    # Fit a polynomial using the actual length of road
    ym_per_px = 30/720
    xm_per_px = 3.7/700
    
    left_fit_meter, left_fit_meter_res, _, _, _ = np.polyfit(lefty*ym_per_px, leftx*xm_per_px, 2, full=True)
    right_fit_meter, right_fit_meter_res, _, _, _ = np.polyfit(righty*ym_per_px, rightx*xm_per_px, 2, full=True)   
    
    # Refresh the fitting parameters for next iterations
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    # Draw transparent lane lines area
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))
    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    out_img = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    
    resultDict = {'left_fit': left_fit_2, 'right_fit':right_fit_2,
    'left_fit_meter':left_fit_meter, 'right_fit_meter':right_fit_meter,
    'left_fit_res':left_fit_res_2, 'right_fit_res':right_fit_res_2,
    'left_fit_meter_res':left_fit_meter_res, 'right_fit_meter_res':right_fit_meter_res}

    return resultDict, out_img
        
'''
Step5: Curvature and Lane Center Deviation
'''
def curvature(in_array, left_fit, right_fit):
    y_eval = np.max(in_array)
    left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    print(left_curverad, right_curverad)
    return left_curverad, right_curverad
    
def curvature_meter(in_array, left_fit_meter, right_fit_meter, ym_per_px = 30/720, xm_per_px = 3.7/700):
    y_eval = np.max(in_array)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_meter[0]*y_eval*ym_per_px + left_fit_meter[1])**2)**1.5) / np.absolute(2*left_fit_meter[0])
    right_curverad = ((1 + (2*right_fit_meter[0]*y_eval*ym_per_px + right_fit_meter[1])**2)**1.5) / np.absolute(2*right_fit_meter[0])
    # Now our radius of curvature is in meters
    print(left_curverad, 'm', right_curverad, 'm')
    return left_curverad, right_curverad
    
def offset(original_image, leftx, rightx, xm_per_px = 3.7/700):
    image_mid = original_image.shape[1]/2
    road_mid = (leftx[-1] + rightx[-1]) / 2
    offset_m =  (image_mid - road_mid) * xm_per_px
    return offset_m
    
'''
Step6: Warp back
       This function draws a filled polynomial defined by 'ploty', 'left_fitx' and
       'right_fitx', it should be called after these THREE parameters are given.
'''
def warpback(original_img, binary_warped_img, Minv, ploty, left_fitx, right_fitx):
    undist = original_img
    warped = binary_warped_img
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))
    print(newwarp[:,:,1].nonzero())
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    lane_area = result
    
    return lane_area
#    plt.imshow(result)

'''
Useful functions: Define some useful functions
'''    
# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self, nSamples):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = np.zeros((nSamples+1,3), dtype=float)  
        #polynomial coefficients for the most recent fit
        self.current_fit = np.zeros(3, dtype=float)  
        #polynomial coefficients for the k-1 step
        self.pre_fit = np.zeros(3, dtype=float)
        #polynomial coefficients for the k-2 step
        self.ppre_fit = np.zeros(3, dtype=float)
        #polynomial coefficients for the most recent fit (meters)
        self.current_fit_meter = np.zeros(3, dtype=float)  
        #polynomial coefficients for the k-1 step (meters)
        self.pre_fit_meter = np.zeros(3, dtype=float)
        #polynomial coefficients for the k-2 step (meters)
        self.ppre_fit_meter = np.zeros(3, dtype=float)
        #radius of curvature of the line in some units
        self.radius_of_curvature = 0
        #average radius of curvature over the last n iterations
        self.best_curvature = np.zeros(nSamples+1, dtype=float)
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None
    
    def range_check(self, element, min_value, max_value):
        if (min_value < element < max_value):
            return True
        else:
            return False
            
    def rationality_check(self, current_value, previous_value, coef):
        diff = current_value - previous_value
        rule_1 = abs(diff) < coef * previous_value
#        print(rule_1)
        rule_2 = current_value * previous_value > 0
        if rule_1 and rule_2:
            pass_flag = True
        else:
            pass_flag = False

        return pass_flag

# Define an algorithm to determine how to update fit parameters
# When normal flag = 1, initialize the normal update process
# When normal flag = 0, use the weighted values which take different
# values of different steps into consideration. Tune the weighted coefficients
# change the influence of last value on the final output value.    
# NOTE: This function should be called by left lane and right lane judgements, respectively !!
#       When calling this function, the parameters current_fit, pre_fit and ppre_fit should represent
#       one of the left side lane or the right side lane.

def updateFitParameters(primary_in, secondary_in, current_out, previous_out, p_previous_out):
    
    global counter, normal_flag
    
    if primary_in[0] * secondary_in[0] > 0:
        counter -= 1
        if counter < 1:
            normal_flag = True
            counter = 0
    else:
        counter += 1
        if counter > 5:
            counter = 5
            normal_flag = False                  
        
    if normal_flag == True:
        print('Use new value')
        current_out = primary_in
        previous_out = current_out
        p_previous_out = previous_out        
    else:
        print('Use old value')
        current_out = 0.9*previous_out + 0.095*p_previous_out + 0.005*primary_in
        previous_out = current_out
        p_previous_out = previous_out
    
    print('counter = ' + str(counter))
                 
    return current_out, previous_out, p_previous_out
            
           

            
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Pipeline Implementation: To test above process functions in following pipeline
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# 1. CAMERA CALIBRATION
# This routin just run once, the camera data is stored in the pickle file
calibration_flag = 0
if calibration_flag != 0:
    calibration_flag = 0
    calibration_img_list = glob.glob('./camera_cal/calibration*.jpg')
    cameraCalibration(calibration_img_list, 9, 6)
    
# If the calibration is done before, just load the camera calibration data sets
with open('cameraData/calibrations.p', 'rb') as f:
    cal_data = pickle.load(f)
    mtx = cal_data['mtx']
    dist = cal_data['dist']

# By default, we launch a blind search routin when finding lanes
blind_search_flag = 1

curvature_dropped = 0
nSamples = 24

#normal_flag = True

image_name_list = glob.glob('./out_jpg/before/foo*.jpg')

# Useful vars for debug
left_fit_array = np.zeros((len(image_name_list), 3), dtype=float)
right_fit_array = np.zeros((len(image_name_list), 3), dtype=float)
left_updated_array = np.zeros((len(image_name_list), 3), dtype=float)
right_updated_array = np.zeros((len(image_name_list), 3), dtype=float)

# Instantiate two lines of the Line class
instance_line_left = Line(nSamples)
instance_line_right = Line(nSamples) 

for idx, image in enumerate(image_name_list):
    print('Current iteration:' + str(idx))
    image = cv2.imread(image)
    
    # 2. CAMERA CALIBRATION: Using calibration parameters to undistort images
    undistort = undistortion(image, mtx, dist)
    # Save undistorted images for testing, Uncommit this when mass processing
#    outfilename = 'undist_test' + str(idx) + '.jpg'
#    cv2.imwrite('./output_images/' + outfilename, undistort)
    
    # 3. THRESHOLDING IMAGES
    binary_image = thresholdingBinary(image)
    # Save thresholding images for testing. Uncommit this when mass processing
#    outfilename = 'binary_test' + str(idx) + '.jpg'
#    cv2.imwrite('./output_images/' + outfilename, binary_image)
    
    # 4. PERSPECTIVE TRANFORM
    warped_img = perspectiveTrans(binary_image)
    
    # 4. FINDING LANE LINES: for each image
    # Use 'blind_search_flag' to switch blind searching and direct searching approach
    if blind_search_flag == 1:
            fit_paras, out_img = slidingWindows(warped_img)
            blind_search_flag = 0
            print('Blind Search Enabled...')
#           print(fit_paras['left_fit'])
    else:
        # Generate x and y values for plotting
        left_fit = fit_paras['left_fit']
        right_fit = fit_paras['right_fit']
#        print('Before direct searching')
#        print(fit_paras['left_fit'])
        fit_paras, out_img = lineFinding(warped_img, left_fit, right_fit)
        print('Direct Search Enabled...')
#        print(fit_paras['left_fit'])
#        print(fit_paras['right_fit'])

    # Load fit parameters
    left_fit = fit_paras['left_fit']
    right_fit = fit_paras['right_fit']
    left_fit_res = fit_paras['left_fit_res']
    right_fit_res = fit_paras['right_fit_res']
    left_fit_meter = fit_paras['left_fit_meter']
    right_fit_meter = fit_paras['right_fit_meter']

    # NOTE: Decide whether update a new fit paras or use the last resonable one
    if idx > 3:
        instance_line_left.current_fit, instance_line_left.pre_fit, instance_line_left.ppre_fit = \
        updateFitParameters(left_fit, right_fit, instance_line_left.current_fit, instance_line_left.pre_fit, instance_line_left.ppre_fit)
        instance_line_right.current_fit, instance_line_right.pre_fit, instance_line_right.ppre_fit = \
        updateFitParameters(right_fit, left_fit, instance_line_right.current_fit, instance_line_right.pre_fit, instance_line_right.ppre_fit)
        left_fit_updated = instance_line_left.current_fit
        right_fit_updated = instance_line_right.current_fit    
        # For meters unit
        instance_line_left.current_fit_meter, instance_line_left.pre_fit_meter, instance_line_left.ppre_fit_meter = \
        updateFitParameters(left_fit_meter, right_fit_meter, instance_line_left.current_fit, instance_line_left.pre_fit, instance_line_left.ppre_fit)
        instance_line_right.current_fit_meter, instance_line_right.pre_fit_meter, instance_line_right.ppre_fit_meter = \
        updateFitParameters(right_fit_meter, left_fit_meter, instance_line_right.current_fit_meter, instance_line_right.pre_fit_meter, instance_line_right.ppre_fit_meter)
        left_fit_meter_updated = instance_line_left.current_fit_meter
        right_fit_meter_updated = instance_line_right.current_fit_meter  
    else:
        left_fit_updated = left_fit
        right_fit_updated = right_fit
        
        left_fit_meter_updated = left_fit_meter
        right_fit_meter_updated = right_fit_meter        
        
    print(left_fit_updated)
    print(right_fit_updated)
    
    left_fit_array[idx] = left_fit
    right_fit_array[idx] = right_fit
    left_updated_array[idx] = left_fit_updated
    right_updated_array[idx] = right_fit_updated                  
#    print('normal flag is: ' + normal_flag)
    
    # Drawing lane lines
    ploty = np.linspace(0, warped_img.shape[0]-1, warped_img.shape[0])
    left_fitx_updated = left_fit_updated[0]*ploty**2 + left_fit_updated[1]*ploty + left_fit_updated[2]
    right_fitx_updated = right_fit_updated[0]*ploty**2 + right_fit_updated[1]*ploty + right_fit_updated[2]
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

# This affects the Lane finding routin!!!!!    
#    fit_paras['left_fit'] = left_fit_updated
#    fit_paras['right_fit'] = right_fit_updated
    # NOTE: Implement a mean value filter integrated with sanity check routin 
    # for the fitting parameters.
#    if idx > nSamples:
#        for k in range(nSamples):
#            if True == instance_line_left.rationality_check(instance_line_left.best_fit[k+1][0],instance_line_left.best_fit[k][0],5)\
#                        and instance_line_right.rationality_check(instance_line_right.best_fit[k+1][0],instance_line_right.best_fit[k][0],5):
#                instance_line_left.best_fit[k] = instance_line_left.best_fit[k + 1]
#                instance_line_left.best_fit[-1] = left_fit
#                instance_line_right.best_fit[k] = instance_line_right.best_fit[k + 1]
#                instance_line_right.best_fit[-1] = right_fit
#                blind_search_flag = 0
#            else:
#                blind_search_flag = 1
#                instance_line_left.best_fit[k] = instance_line_left.best_fit[k + 1]
#                instance_line_left.best_fit[-1] = instance_line_left.best_fit[k + 1]
#                instance_line_right.best_fit[k] = instance_line_right.best_fit[k + 1]
#                instance_line_right.best_fit[-1] = instance_line_right.best_fit[k + 1]                            
#    else:
#        instance_line_left.best_fit[idx] = left_fit
#        instance_line_right.best_fit[idx] = right_fit
#    
#    instance_line_left.current_fit = np.mean(instance_line_left.best_fit, axis=0)      
#    instance_line_right.current_fit = np.mean(instance_line_right.best_fit, axis=0)       
#    
#    filtered_fit_l = instance_line_left.current_fit
#    filtered_fit_r = instance_line_right.current_fit

    # NOTE: fit parameters should be stored in THREE variables, k, k-1, k-2 
    
    
    # 5. CURVATUREs
    curvature_m_left, curvature_m_right = curvature_meter(ploty, left_fit_meter_updated, right_fit_meter_updated)
    # When TWO curvatures have unreasonable difference, we have to choose one
    # Cause the practical curvature of two lanes are nearly equal
    if left_fit_res < right_fit_res:
        print('Left was chosen!')
        if left_fit_updated[0]<0 and right_fit_updated[0]<0:
            curvature_m_left, curvature_m_right = curvature_m_left, curvature_m_left + 3.7
        elif left_fit_updated[0]>0 and right_fit_updated[0]>0:
            curvature_m_left, curvature_m_right = curvature_m_left + 3.7, curvature_m_left
        else:
            curvature_m_left, curvature_m_right = curvature_m_left, curvature_m_right           
    else:
        print('Right was chosen!')
        if left_fit_updated[0]<0 and right_fit_updated[0]<0:
            curvature_m_left, curvature_m_right = curvature_m_right, curvature_m_right + 3.7
        elif left_fit_updated[0]>0 and right_fit_updated[0]>0:
            curvature_m_left, curvature_m_right = curvature_m_right + 3.7, curvature_m_right
        else:
            curvature_m_left, curvature_m_right = curvature_m_left, curvature_m_right
            
        
#    print(curvature_m_left, curvature_m_right)
    
    # NOTE: Implement a digital mean value filter (nSamples = 31)   
    if idx > nSamples:
        for jdx in range(nSamples):
            instance_line_left.best_curvature[jdx] = instance_line_left.best_curvature[jdx+1]
            if (instance_line_left.range_check(curvature_m_left, 20, 1e4)) and (instance_line_left.rationality_check(curvature_m_left, instance_line_left.best_curvature[-1], 1)):
                instance_line_left.best_curvature[-1] = curvature_m_left
#                normal_flag = True
            else:
#                normal_flag = False
                blind_search_flag = 1
                
            instance_line_right.best_curvature[jdx] = instance_line_right.best_curvature[jdx+1]
            if (instance_line_right.range_check(curvature_m_right, 20, 1e4)) and (instance_line_right.rationality_check(curvature_m_right, instance_line_right.best_curvature[-1], 1)):
                instance_line_right.best_curvature[-1] = curvature_m_right
#                normal_flag = True
            else:
#                normal_flag = False
                blind_search_flag = 1
    else:
        instance_line_left.best_curvature[idx] = curvature_m_left
        instance_line_right.best_curvature[idx] = curvature_m_right

    max_left, max_right = np.max(instance_line_left.best_curvature), np.max(instance_line_right.best_curvature)
    min_left, min_right = np.min(instance_line_left.best_curvature), np.min(instance_line_right.best_curvature)
    
    instance_line_left.radius_of_curvature = (np.sum(instance_line_left.best_curvature) - max_left - min_left) / (nSamples + 1- 2)
    instance_line_right.radius_of_curvature = (np.sum(instance_line_right.best_curvature) - max_right - min_right) / (nSamples + 1 - 2)
    
    curvature_m_est = instance_line_left.radius_of_curvature
    
#    if (100<curvature_m_left<10000) and (100<curvature_m_right<10000) and (abs(curvature_m_left - curvature_m_right) < 80):
#        instance_line_left.radius_of_curvature = curvature_m_left
#        instance_line_right.radius_of_curvature = curvature_m_right
#        curvature_m_est = 0.5 * (instance_line_left.radius_of_curvature + instance_line_right.radius_of_curvature)
#        instance_line_left.best_curvature += curvature_m_est
#        instance_line_right.best_curvature += curvature_m_est
#    else:
#        curvature_dropped += 1
#        instance_line_left.radius_of_curvature = 0.5 * instance_line_left.best_curvature / (idx + 2 - curvature_dropped)\
#        + 0.5 * curvature_m_left
#        instance_line_right.radius_of_curvature = 0.5 * instance_line_right.best_curvature / (idx + 2 - curvature_dropped)
#        + 0.5 * curvature_m_left
#        curvature_m_est = 0.5 * (instance_line_left.radius_of_curvature + instance_line_right.radius_of_curvature)
        
         
    
    
    with open('cameraData/perspective.p', 'rb') as f:
        perspect_data = pickle.load(f)
        M = perspect_data['M']
        Minv = perspect_data['Minv']

    # 6. MERGE IMAGE & LANEs
    lane_shaded = warpback(undistort, warped_img, Minv, ploty, left_fitx_updated, right_fitx_updated)
    offset_m = offset(undistort, left_fitx_updated, right_fitx_updated)
    
    # DRAW TEXT 
    display_curvature = '[ Current Curvature is: ' + "{:03.2f}".format(curvature_m_est) + 'm ]'
    display_deviation = '[ Current Deviation is: ' + "{:03.2f}".format(offset_m) + 'm]'
    warning = '[ Deviation Warning! ' + "{:03.2f}".format(offset_m) + 'm]'
    cv2.putText(lane_shaded, display_curvature, (250,60),cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255, 0), thickness = 2)
    if abs(offset_m) > 0.5:
        cv2.putText(lane_shaded, warning, (250,100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), thickness = 2)
    else:
        cv2.putText(lane_shaded, display_deviation, (250,100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), thickness = 2)

    
    # Plot the result
    # WARNING: Shutdown the display when mass processing images !!
#    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
#    f.tight_layout()
#    
#    ax1.imshow(cv2.cvtColor(lane_shaded, cv2.COLOR_BGR2RGB))
#    ax1.set_title('Original Image', fontsize=40)
#    
#    ax2.imshow(out_img)
#    plt.plot(left_fitx, ploty, color='yellow')
#    plt.plot(right_fitx, ploty, color='yellow')  
#    plt.xlim(0, 1280)
#    plt.ylim(720, 0)
#    ax2.set_title('Pipeline Result', fontsize=40)
#    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)    
    
    # Store lane images in the output folder
    out_folder = './out_jpg/after/'
    out_name = 'out-' + str(idx+1).rjust(4,'0') + '.jpg'
    cv2.imwrite(out_folder+out_name, lane_shaded)
    
    print('...........')
    
f, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(20, 9))
f.tight_layout()
ax1.plot(np.linspace(0,left_fit_array.shape[0]-1, left_fit_array.shape[0]), left_fit_array[:,0])
ax1.set_title('left_fit')
ax2.plot(np.linspace(0,left_fit_array.shape[0]-1, left_fit_array.shape[0]), right_fit_array[:,0])
ax2.set_title('right_fit')
ax3.plot(np.linspace(0,left_fit_array.shape[0]-1, left_fit_array.shape[0]), left_updated_array[:,0])
ax3.set_title('left_fit_updated')
ax4.plot(np.linspace(0,left_fit_array.shape[0]-1, left_fit_array.shape[0]), right_updated_array[:,0])
ax4.set_title('right_fit_updated')