import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import prepro as pp
import histograma as his


def main():

    img=cv.imread('./1.jpg') #imagem original
    img=pp.resized(img,50)
    cv.imshow('Original',img) 
    
    gray=pp.prePro(img)
    cv.imshow('Escala de cinza',gray)
    
    filtrada=pp.filtro("pxy",gray)
    cv.imshow("aaa",filtrada)

    cv.waitKey(0)
    cv.destroyAllWindows()

main()
