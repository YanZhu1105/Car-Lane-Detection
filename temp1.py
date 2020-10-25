import matplotlib.pyplot as plt
import cv2
import numpy as np


image = cv2.imread("170927_064028573_Camera_5_bin.png", cv2.IMREAD_GRAYSCALE)
print(np.unique(image))
plt.imshow(image)
plt.show()