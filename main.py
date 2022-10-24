import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import prepro as pp
import histograma as his

def otsu(img):
    sum = 0
    sumB = 0
    wB = 0 #peso do fundo
    wF = 0 #peso da frente
    varMax = 0
    threshold = 0 #limite da limiarização

    hist = his.histogramaOtsu(img)#faz o histograma da imagem
    total = img.shape[0] * img.shape[1]#numero de linhas x colunas (total de pixels)

    for i in range(0, 256):
        sum += i * hist[i]
        


    for i in range(0, 256):
        wB += hist[i]
        if wB == 0:
            continue
        wF = total - wB #peso da frente é o total de pixels menos o peso do fundo
        if wF == 0:
            break
        sumB += i * hist[i]
        mB = sumB / wB #calculando media do fundo
        mF = (sum - sumB) / wF #calculando media da frente
        varBetween = wB * wF * (mB - mF) * (mB - mF) #calculo da variancia
        if varBetween > varMax:#a variancia que for maior que max determina o ton de limearização
            varMax = varBetween
            threshold = i#valor limite

    return threshold #retorna o limite



def main():
    img=cv.imread('./1.jpg') #imagem original
    img=pp.resized(img,50)
    cv.imshow('Original',img) 
    

    gray=pp.prePro(img)
    cv.imshow('Escala de cinza',gray)
    
#passa alta
    prewittx=np.array([[-1, -1,  -1],
                   [0,  0, 0],
                    [1, 1,  1]])

    prewitty=np.array([[-1, 0,  1],
                   [-1,  0, 1],
                    [-1, 0,  1]])

    laplace2=np.array([[-1, -1, -1],
                   [-1,  8, -1],
                    [-1, -1,  -1]])

    filtradax=pp.filtro(prewittx,gray)
    cv.imshow('prewitt X',filtradax)

    filtraday=pp.filtro(prewitty,gray)
    cv.imshow('prewitt Y',filtraday)

    filtradaxy = cv.addWeighted(filtradax, 0.5, filtraday, 0.5, 0)
    cv.imshow('prewitt XY',filtradaxy)

    filtrada=pp.filtro(laplace2,gray)
    cv.imshow('laplaciana l2',filtrada)

    
#passa baixa
    media=np.array([[1/9, 1/9,  1/9],
                   [1/9,  1/9, 1/9],
                    [1/9, 1/9,  1/9]])
    
    gaussiano=np.array([[1/16, 2/16,  1/16],
                   [2/16,  4/16, 2/16],
                    [1/16, 2/16,  1/16]])

    #kernel = np.ones((3,3), np.float32) / 9

    filtrada=pp.filtro(gaussiano,gray)
    cv.imshow('gaussiano',filtrada)

    filtrada=pp.filtro(media,gray)
    cv.imshow('media',filtrada)

    #(thresh, bin) = cv.threshold(filtrada, otsu(gray), 255, cv.THRESH_BINARY) #binarizando com otsu como limite
    #cv.imshow('Binarizada Otsu', bin)


    #(thresh, bin) = cv.threshold(filtrada, 120, 255, cv.THRESH_BINARY)#binarização com limite fixo para comparar
    #cv.imshow('Binarizada Fixa', bin)

    #plt.show()#mostra o histograma

    

    cv.waitKey(0)
    cv.destroyAllWindows()

main()
