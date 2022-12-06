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
winSize = (64,64)#tamanho da janela
blockSize = (16,16)#tamanho dos blocos/subimagens
blockStride = (8,8)#distancia entre os blocos
cellSize = (8,8)#tamanho da celula dos histogramas
nbins = 9#valores do histograma
derivAperture = 1#abertura do gradient
winSigma = 4.#suavização gaussiana
histogramNormType = 0#tipo de normalização usada
L2HysThreshold = 2.0000000000000001e-01#limiar de normalização
gammaCorrection = 0#Correção da gama da imagem
nlevels = 64#quantidade de niveis usado para a orientação

hog = cv.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels)


winStride = (8,8)
padding = (8,8)
locations = ((10,20),)

hist = hog.compute(image,winStride,padding,locations)
#hist=cv.normalize(hist, hist, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)

hist2 = hog.compute(image2,winStride,padding,locations) 
#hist2=cv.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)

#print(hist)
#print(hist2)

compara = cv.compareHist(hist, hist2, 0)

print(compara)

cv.waitKey(0)
cv.destroyAllWindows()