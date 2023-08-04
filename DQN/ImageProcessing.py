import numpy as np


def clipping(image):
    return image[31:193, 8:152, :]


def grayscale(image):
    return np.mean(image, axis=2, keepdims=False)


def normalization(image):
    return image / 256
