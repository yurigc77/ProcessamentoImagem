import cv2 as cv
import matplotlib.pyplot as plt

def histogramaOtsu(img):
    hist = cv.calcHist([img], [0], None, [256], [0, 256])
    plt.figure()
    plt.title("Histograma")
    plt.xlabel("Tons")
    plt.ylabel("Número de Pixels")
    plt.plot(hist)
    plt.xlim([0, 256])
    return hist