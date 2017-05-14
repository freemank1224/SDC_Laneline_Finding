# -*- coding: utf-8 -*-
"""
Warp image

Created on Sat Apr  1 22:12:36 2017

@author: dyson
"""

from functionLib import *

# Using this function to get a warpped image
 
image_name_list = glob.glob('./output_images/undist/undist_test*.jpg')

for idx, image in enumerate(image_name_list):
    image = cv2.imread(image)
    binary_image = thresholdingBinary(image)
    warped_image = perspectiveTrans(binary_image)
    outfilename = 'warped_test' + str(idx) + '.jpg'
    print(outfilename)
    cv2.imwrite('./output_images/warped/' + outfilename, binary_image)
#    warped_img = perspectiveTrans(binary_image)    
    
    # Plot the result
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    
    ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax1.set_title('Original Image', fontsize=40)
    
    ax2.imshow(warped_image, cmap='gray')
    ax2.set_title('Warped Image', fontsize=40)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)