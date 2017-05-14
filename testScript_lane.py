# -*- coding: utf-8 -*-
"""
Fit polynomial

Created on Sat Apr  1 22:08:44 2017

@author: dyson
"""

from functionLib import *

blind_search_flag = 1

image = './output_images/undist/undist_test2.jpg'


image = cv2.imread(image)
binary_image = thresholdingBinary(image)
warped_image = perspectiveTrans(binary_image)
#    outfilename = 'warped_test' + str(idx) + '.jpg'
#    print(outfilename)
#    cv2.imwrite('./output_images/warped/' + outfilename, binary_image)


slidingWindows(warped_image)
