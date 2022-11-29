# import required libraries
import cv2 as cv
import numpy as np
import prepro as pp


# define the function to compute MSE between two images
def mse(img1, img2):
   h, w = img1.shape
   img2=pp.sameSize(img1,img2)
   diff = cv.subtract(img1, img2)
   err = np.sum(diff**2)
   mse = err/(float(h*w))
   return mse, diff

def main():
    # load the input images
    img1 = cv.imread('./ass.jpg')
    img2 = cv.imread('./assLuis.jpg')

    # convert the images to grayscale
    img1 = pp.prePro(img1)
    img2 = pp.prePro(img2)

    img1=pp.filtro("l2",img1)
    img2=pp.filtro("l2",img2)

    error, diff = mse(img1, img2)
    print("Image matching Error between the two images:",error)

    cv.imshow("difference", diff)
    cv.waitKey(0)
    cv.destroyAllWindows()

main()