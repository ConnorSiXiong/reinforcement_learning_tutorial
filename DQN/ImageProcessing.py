import numpy as np


def clipping(image):
    return image[31:193, 8:152, :]


def grayscale(image):
    return np.mean(image, axis=2, keepdims=False)


def down_sampling(image):
    return image[::2, ::2]


def normalization(image):
    return image / 256


def process_image(image):
    image = clipping(image)
    image = grayscale(image)
    image = down_sampling(image)
    return normalization(image)
