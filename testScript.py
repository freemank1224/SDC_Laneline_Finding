# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 14:00:25 2017

Test Script for Project 4

@author: Dyson
"""
from functionLib import *

with open('./cameraData/calibrations.p','rb') as f: 
    cal_data = pickle.load(f)
    mtx = cal_data['mtx']
    dist = cal_data['dist']

image_name_list = glob.glob('./test_images/test*.jpg')

for idx, image in enumerate(image_name_list):
    image = cv2.imread(image)
    #print(idx, image)
    undistort = undistortion(image, mtx, dist)
    outfilename = 'undist_test' + str(idx) + '.jpg'
    print(outfilename)
    cv2.imwrite('./output_images/undist/' + outfilename, undistort)
    
    #Visualize undistortion Obly for testing!
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24,10))
    f.tight_layout()
    ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax1.set_title('Original Image', fontsize=30)
    ax2.imshow(cv2.cvtColor(undistort, cv2.COLOR_BGR2RGB))
    ax2.set_title('Undistorted Image', fontsize=30)
    
    
