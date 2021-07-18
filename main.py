import numpy as np
import cv2 as cv
from preprocessing import PreProcessing
from kernel import Poisson

if __name__ == '__main__':
    src = cv.imread('src.jpeg',cv.IMREAD_GRAYSCALE)
    dst = cv.imread('dst.jpeg', cv.IMREAD_GRAYSCALE)
    pre = PreProcessing()
    pre.select(src, dst)

    retImg = Poisson.seamlessClone(src, dst, pre.selectedMask, pre.selectedPoint, Poisson.NORMAL_CLONE)
    cv.imshow('result', retImg)
    cv.waitKey(0)
    cv.destroyAllWindows()
    cv.imwrite('result.jpeg', retImg)
