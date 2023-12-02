#!/bin/env python3

import cv2 as cv
import numpy as np
import time
import sys

np.set_printoptions(threshold=sys.maxsize)

def im2col(img, kernels_array):
    # transform kernels to rows
    new_kernels_array_shape = list(kernels_array.shape)
    new_kernels_array_shape[1] *= new_kernels_array_shape.pop()
    new_kernels_array = np.zeros(tuple(new_kernels_array_shape))
    for ind, elem in enumerate(kernels_array):
        new_elem = list(elem[0])
        new_elem.extend(elem[1])
        new_elem.extend(elem[2])

        new_kernels_array[ind] = new_elem

    output_image_shape = (img.shape[0] - kernels_array[0].shape[0], img.shape[1] - kernels_array[0].shape[1], kernels_array.shape[0])
    output_image = np.zeros(output_image_shape)
    patch_matrix_shape = [new_kernels_array_shape[1], output_image_shape[0] * output_image_shape[1]]
    patch_matrix = np.zeros(patch_matrix_shape)
    for k in (range(img.shape[2])):
        # transform image to column:
        for x, y in np.ndindex(output_image_shape[:-1]):
            for i, j in np.ndindex(kernels_array[0].shape):
                patch_matrix[i * (kernels_array.shape[1]) + j][x * (img.shape[1] - kernels_array[0].shape[0]) + y] = img[x+i][y+j][k]

        # matrix multiplication
        for m in range(kernels_array.shape[0]):
            result_matrix = np.matmul(new_kernels_array[m].transpose(), patch_matrix)
            for x, y in np.ndindex(output_image_shape[:-1]):
                output_image[x][y][m] += result_matrix[x * output_image_shape[1] + y]

    return output_image


def base(img: np.array, kernels_array):
    output_img_shape = list(img.shape)
    output_img_shape.pop()

    for id, elem in enumerate(output_img_shape):
        output_img_shape[id] -= kernels_array.shape[id]

    output_img_shape.append(kernels_array.shape[0])

    output_img = np.zeros(output_img_shape)

    for k in range(img.shape[2]):
        for x, y, m in np.ndindex(tuple(output_img_shape)):
            for i, j in np.ndindex(kernels_array[0].shape):
                output_img[x][y][m] += img[x+i][y+j][k] * kernels_array[m][i][j]

    return output_img

img = cv.imread("../lab1/F1.jpg")

kernels = np.array([[
           [0, 1, 0],
           [1, 2, 1],
           [0, 1, 0]],
           [[0, -1, 0],
           [-1, -2, -1],
           [0, -1, 0]],
           [[1, 2, 3],
            [3, 2, 1],
            [0, 1, 0]]])

start = time.time()
base_img = base(img, kernels)
print("base func time =", time.time() - start)

start = time.time()
new_img = im2col(img, kernels)
print("im2col func time =", time.time() - start)

max_val = 0
for x, y, m in np.ndindex(new_img.shape):
    max_val = max(max_val, (new_img[x][y][m] - base_img[x][y][m])**2)

print("max difference between base variant and im2col =", max_val)

 