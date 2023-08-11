import numpy as np
import torch
from config import device


def clipping(image):
    return image[31:193, 8:152, :]


def grayscale(image):
    return np.mean(image, axis=2, keepdims=False)


def down_sampling(image):
    return image[::2, ::2]


def padding(image):
    desired_shape = (84, 84)
    pad_rows = (desired_shape[0] - image.shape[0]) // 2
    pad_cols = (desired_shape[1] - image.shape[1]) // 2
    return np.pad(image, ((pad_rows + 1, pad_rows), (pad_cols, pad_cols)), mode='constant', constant_values=0)


def normalization(image):
    return np.ascontiguousarray(image, dtype=np.float32) / 255.0


def process_image(image):
    image = clipping(image)
    image = grayscale(image)
    image = down_sampling(image)
    image = normalization(image)
    # image = padding(image)

    return torch.tensor(image, dtype=torch.float32, device=device).unsqueeze(0)
