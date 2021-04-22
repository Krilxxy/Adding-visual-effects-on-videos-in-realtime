import cv2
import numpy as np

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eyeCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

if faceCascade.empty() :
    raise IOError('Unable to load the face cascade classifier xml file.')
if eyeCascade.empty() :
    raise IOError('Unable to load the eye cascade classifier xml file.')

cap = cv2.VideoCapture(0)
foxFilter_img = cv2.imread('./resources/sunglasses.png')
scaling_factor = 0.0387358024691358
dst = 37

print(foxFilter_img.shape)
newimg = cv2.resize(foxFilter_img,None, fx=scaling_factor, fy=scaling_factor,interpolation=cv2.INTER_AREA)
cv2.imshow('img', newimg)
cv2.waitKey() 