# from sklearn.datasets import load_digits
# import matplotlib.pyplot as plt
#
# digits = load_digits()
# print(digits.data.shape)
# plt.gray()

# plt.matshow(digits.images[3])
# plt.show()

import cv2
import numpy as np

def _resize_image(image, width, height):
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
from sklearn import datasets

import matplotlib.pyplot as plt

# Load the digits dataset
digits = datasets.load_digits()

# Display the last digit
plt.figure(1, figsize=(3, 3))

width = 480
height = 480
tensor= digits.images[2]
image = _resize_image(np.asarray(tensor), width, height)
plt.imshow(digits.images[2], cmap=plt.cm.gray_r, interpolation="nearest")
plt.show()

