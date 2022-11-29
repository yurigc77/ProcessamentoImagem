from __future__ import print_function
from __future__ import division
import cv2 as cv
import numpy as np
import argparse
import prepro as pp


    
yuri=cv.imread('./ass.jpg')
yuri2=cv.imread('./ass2.jpg')
maria=cv.imread('./assMaria.png')
luis=cv.imread('./assLuis.jpg')


yuri = cv.cvtColor(yuri, cv.COLOR_BGR2HSV)
yuri2 = cv.cvtColor(yuri2, cv.COLOR_BGR2HSV)
maria = cv.cvtColor(maria, cv.COLOR_BGR2HSV)
luis = cv.cvtColor(luis, cv.COLOR_BGR2HSV)

h_bins = 50
s_bins = 60
histSize = [h_bins, s_bins]

# hue varies from 0 to 179, saturation from 0 to 255
h_ranges = [0, 180]
s_ranges = [0, 256]
ranges = h_ranges + s_ranges # concat lists

# Use the 0-th and 1-st channels
channels = [0, 1]

histyuri = cv.calcHist([yuri], channels, None, histSize, ranges, accumulate=False)
cv.normalize(histyuri, histyuri, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)

histyuri2 = cv.calcHist([yuri2], channels, None, histSize, ranges, accumulate=False)
cv.normalize(histyuri2, histyuri2, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)

histmaria = cv.calcHist([maria], channels, None, histSize, ranges, accumulate=False)
cv.normalize(histmaria, histmaria, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)

histluis = cv.calcHist([luis], channels, None, histSize, ranges, accumulate=False)
cv.normalize(histluis, histluis, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)


#for compare_method in range(4):
compare_method=0
yuri_yuri = cv.compareHist(histyuri, histyuri, compare_method)
yuri_yuri2 = cv.compareHist(histyuri, histyuri2, compare_method)
yuri_maria = cv.compareHist(histyuri, histmaria, compare_method)
yuri_luis = cv.compareHist(histyuri, histluis, compare_method)
print('comparando:yuri, yuri2, maria, luis :\n',\
    yuri_yuri, '/', yuri_yuri2, '/', yuri_maria, '/', yuri_luis)