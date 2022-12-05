import cv2 as cv
import prepro as pp
#importing required libraries
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure
import matplotlib.pyplot as plt


image = cv.imread("./ass.jpg",0)
image2 = cv.imread("./assLuis.jpg",0)

image=pp.filtro("px",image)
image2=pp.filtro("px",image2)

#variaveis de hog
winSize = (64,64)
blockSize = (16,16)
blockStride = (8,8)
cellSize = (8,8)
nbins = 9
derivAperture = 1
winSigma = 4.
histogramNormType = 0
L2HysThreshold = 2.0000000000000001e-01
gammaCorrection = 0
nlevels = 64

hog = cv.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels)

#compute(img[, winStride[, padding[, locations]]]) -> descriptors

winStride = (8,8)
padding = (8,8)
locations = ((10,20),)

hist = hog.compute(image,winStride,padding,locations)
hist=cv.normalize(hist, hist, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)

hist2 = hog.compute(image2,winStride,padding,locations)
hist2=cv.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)

#print(hist)
#print(hist2)

compara = cv.compareHist(hist, hist2, 0)

print(compara)

cv.waitKey(0)
cv.destroyAllWindows()