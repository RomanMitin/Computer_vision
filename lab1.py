#!/bin/env python3

import cv2 as cv
import numpy as np

output_to_file = False

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

if(output_to_file):
    cv.imwrite("Face image.jpg", img)
    cv.imwrite('Edges_image.jpg', edges_image) 
    cv.imwrite('Dilate_image.jpg', dilate_image) 
else:
    # Вывод полученных изображений
    cv.imshow('Face image', img) 
    cv.imshow('Edges image', edges_image) 
    cv.imshow('Dilate image', dilate_image) 
    cv.waitKey(0) 


