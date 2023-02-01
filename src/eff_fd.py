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
        return img[y:y+M,x:x+M]

    @staticmethod
    def _resize(img):
        M = 2 ** int(math.log(img.shape[0], 2))
        return cv2.resize(img, (M,M))

    @staticmethod
    def _getMass(img, s):
        height, width = img.shape[:2]
        cell_height = height // s
        cell_width = width // s
        reshaped_image = img.reshape(s, cell_height, s, cell_width).swapaxes(1,2)
        reshaped_image = reshaped_image.astype(np.uint16)
        grid_max = np.amax(reshaped_image, axis=(2, 3))
        grid_min = np.amin(reshaped_image, axis=(2, 3))
        return np.sum(grid_max - grid_min + 1)

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
    def getHausdorffDimension(img, plot=False):
        img = EffFD._crop(img)
        img = EffFD._resize(img)

        if (plot):
            plt.imshow(img, cmap='magma')
            plt.show()

        masses = []
        M = img.shape[0]
        s = 2
        
        while (s < M):
            masses += [(s,EffFD._getMass(img, s))]
            s *= 2
        
        dim = EffFD._fit(masses, plot)
        return dim