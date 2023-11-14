#!/bin/env python3

import cv2 as cv
import numpy as np

output_to_file = True

# Считывание изображения
img = cv.imread("image.jpg")
gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY) 

# Детектирование лица на изображении
haar_cascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
faces_rect = haar_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=9) 

for (x, y, w, h) in faces_rect: 
    y -= 50
    w += w // 10
    h += 60
    cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), thickness=2) 

    if(output_to_file):
        cv.imwrite("Detected_face.jpg", img)
    else:
        cv.imshow('Detected face', img) 

    img = img[y:y+h, x:x+w]
    gray_img = gray_img[y:y+h, x:x+w]

# Получение границ на изображении
edges_image = cv.Canny(gray_img, 100, 200)

# Нахождение угловых точек на изображении и добавление их на изображение с границами
corners = cv.cornerHarris(gray_img, 2, 3, 0.04)
edges_image[corners > 0.01 * corners.max()] = 255

# Применение морфологической операции наращивания
kernel = np.ones((5,5),np.uint8)
dilate_image = cv.dilate(edges_image, kernel)

smooth_image = cv.GaussianBlur(dilate_image, [5,5], 1)

image_M = np.zeros((len(img), len(img[0])))

for i, row in enumerate(img):
    for j, pixel in enumerate(row):
        image_M[i][j] = (float(pixel[0]) + pixel[1] + pixel[2]) / (255 * 3)

image_F1 = cv.GaussianBlur(img, [7,7], 1)

image_F2 = cv.cvtColor(img, cv.COLOR_RGB2HSV)

for i, row in enumerate(image_F2):
    for j, pixel in enumerate(row):
        image_F2[i][j][1] *= 1.2
        image_F2[i][j][1] = np.int8(min(image_F2[i][j][1], 255))

image_F2 = cv.GaussianBlur(image_F2, [7,7], 1)

image_final = np.zeros((len(img), len(img[0]), 3))

for i, row in enumerate(image_F2):
    for j, pixel in enumerate(row):
        for c in range(0, 3):
            image_final[i][j][c] = np.int8(min(image_M[i][j] * image_F2[i][j][c] + (1 - image_M[i][j]) * image_F1[i][j][c], 255))

if(output_to_file):
    cv.imwrite("Face image.jpg", img)
    cv.imwrite('Edges_image.jpg', edges_image) 
    cv.imwrite('Dilate_image.jpg', dilate_image) 
    cv.imwrite('Smooth_image.jpg', smooth_image) 
    cv.imwrite('M.jpg', image_M) 
    cv.imwrite('F1.jpg', image_F1) 
    cv.imwrite('F2.jpg', image_F2) 
    cv.imwrite('Final_image.jpg', image_final) 
else:
    pass
    # Вывод полученных изображений
    cv.imshow('Face image', img) 
    cv.imshow('Edges image', edges_image) 
    cv.imshow('Dilate image', dilate_image) 
    cv.imshow('Smooth image', smooth_image) 

    cv.imshow('M image', image_M) 
    cv.imshow('F1 image', image_F1) 
    cv.imshow('hsv image', image_F2) 
    cv.imshow('image_final image', image_final) 
    cv.waitKey(0) 


