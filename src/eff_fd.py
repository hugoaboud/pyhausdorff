import cv2
import math
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Sarkar, N., & Chaudhuri, B. B. (1992).
# An efficient approach to estimate fractal dimension of textural images.
# Pattern Recognition, 25(9), 1035–1041.
# doi:10.1016/0031-3203(92)90066-r 

class EffFD:

    @staticmethod
    def _crop(img):
        M = min(img.shape[0], img.shape[1])
        h = img.shape[0]
        w = img.shape[1]
        y = int((h-M)/2)
        x = int((w-M)/2)
        return img[y:y+M,x:x+M], M

    @staticmethod
    def _getMass(img, M, s):
        mass = 0
        for y in range(0,img.shape[0],s):
            for x in range(0,img.shape[1],s):
                box = img[y:y+s,x:x+s]
                min = np.min(box)
                max = np.max(box)
                mass += max-min+1
        return mass

    @staticmethod
    def _fit(masses, plot=False):
        x = [math.log(m[0]) for m in masses]
        y = [math.log(m[1]) for m in masses]
        fit = np.polyfit(x, y, 1)
        yfit = [px*fit[0] + fit[1] for px in x]
        if (plot):
            plt.plot([m[0] for m in masses],[m[1] for m in masses])
            plt.show()
            plt.plot(x, y)
            plt.plot(x, yfit)
            plt.show()
        return fit[0]

    @staticmethod
    def getHausdorffDimension(img):
        # plt.imshow(img, cmap='magma')
        # plt.show()
        img, M = SDBC._crop(img)
        masses = []
        s = int(M/2)
        while (s > 1):
            masses += [(M/s,SDBC._getMass(img, M, s))]
            s = int(s/2)
        # S = range(2, int(M/2)+1, 1)
        # masses = [(M/s,SDBC._getMass(img, M, s)) for s in S]
        dim = SDBC._fit(masses)
        return dim