import cv2 as cv
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils


def filtro(kernel,img):

    if kernel == "px":#prewittx
        kernel=np.array([[-1, -1,  -1],
                   [0,  0, 0],
                    [1, 1,  1]])
    elif kernel=="py":#prewitty
        kernel=np.array([[-1, 0,  1],
                   [-1,  0, 1],
                    [-1, 0,  1]])
    elif kernel=="pxy":#prewittxy
        filtradax=filtro("px",img)
        filtraday=filtro("py",img)
        img = cv.addWeighted(filtradax, 0.5, filtraday, 0.5, 0)
        return img
    elif kernel=="l2":#laplace 2
        kernel=np.array([[-1, -1, -1],
                   [-1,  8, -1],
                    [-1, -1,  -1]])
    elif kernel=="m":#media
        kernel=np.array([[1/9, 1/9,  1/9],
                   [1/9,  1/9, 1/9],
                    [1/9, 1/9,  1/9]])
    elif kernel=="g":#gaussiano
        kernel=np.array([[1/16, 2/16,  1/16],
                   [2/16,  4/16, 2/16],
                    [1/16, 2/16,  1/16]])
    else:
        print("digite uma opção valida")
        return img

    img = cv.filter2D(src=img, ddepth=-1, kernel=kernel)
    return img

def findContours(img):
    cnts = cv.findContours(img.copy(), cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    return cnts



def edgeDetection(img):
    edged = cv.Canny(img, 50, 100)
    edged = cv.dilate(edged, None, iterations=1)
    edged = cv.erode(edged, None, iterations=1)
    return edged


def resized(img,scale):
    scale_percent = scale# percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    return cv.resize(img, dim, interpolation = cv.INTER_AREA)

def sameSize(img1,img2):
    width = int(img1.shape[1])
    height = int(img1.shape[0])
    dim = (width, height)
    #h, w = yuri.shape
    img2 = cv.resize(img2,dim)

    return img2

def prePro(img):
   
    img=cv.cvtColor(img,cv.COLOR_BGR2GRAY) #imagem cinza
   
    img=cv.GaussianBlur(img,(7,7),0) #blur pra melhorar a imagem
    
    return img


