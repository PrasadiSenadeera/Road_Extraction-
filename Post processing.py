# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 16:43:17 2020

To post process the extracted road features from U_net image
segmentation architecture

@author: KUHPT Senadeera
As a part of Master thesis for complete the of Geosptial Technologies 
"""


import numpy as np 
import cv2
import pandas as pd
from skimage.io import imshow
import matplotlib.pyplot as plt

width=[]
length=[]
C_area=[]
B_area=[]
Rectangularity=[]
A_ratio=[]

#read image to be post processed

image = cv2.imread('2.png')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
retr , thresh = cv2.threshold(gray_image, 127, 255, 0)
contours, hier = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# create emtpy mask
mask = np.zeros(image.shape[:2], dtype=image.dtype)

# draw all contours larger than 100 on the mask
for c in contours:
    if cv2.contourArea(c) >100:
        print(cv2.contourArea(c))
        x, y, w, h = cv2.boundingRect(c)
        cv2.drawContours(mask, [c], 0, (1), -1)

# apply the mask to the original image
result_1 = cv2.bitwise_and(image,image, mask= mask)

#save image (not mandotary)
cv2.imwrite("result_1.png", result_1)


#img = cv2.imread('result_1.png')
imgray = cv2.cvtColor(result_1, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 127, 255, 0)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#create MABR and collect aspect ratio and rectagualrity

for c in contours:
    
    #minimum area rectangle
    rect = cv2.minAreaRect(c)  
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    # Draw contours for box
    #cv2.drawContours(result_1, [box], 0, (0,0,255), 2)
    
    #find variables
    length.append(rect[1][0])
    width.append(rect[1][1])
    a=cv2.contourArea(c)      
    C_area.append(a)


df = pd.DataFrame()

df['width']=width
df['length']=length
df['contour area']=C_area

for i in df:
    B_area=df['width']*df['length']
    B_area.append(B_area)
df['B_area']=B_area

for i in df:
    A_ratio=df['length']/df['width']
    A_ratio.append(A_ratio)
df['A_ratio']=A_ratio

for i in df:
    Rectangularity=df['contour area']/df['B_area']
    Rectangularity.append(Rectangularity)
df['Rectangularity']=Rectangularity

cv2.drawContours(result_1, contours, -1, (0,255,0), 1)
cv2.imwrite("result_2.png", result_1)


image = cv2.imread('result_2.png')
imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
imshow(image)
plt.show()

#Apply morphological operations
kernel = np.ones((5,5),np.uint8)
kernel2 = np.ones((5,5),np.uint8)

#opening = cv2.morphologyEx(imgray, cv2.MORPH_OPEN, kernel)
#cv2.imwrite("opening.png", opening)

closing = cv2.morphologyEx(imgray, cv2.MORPH_CLOSE, kernel2)
closing = cv2.morphologyEx(closing, cv2.MORPH_CLOSE, kernel2)
cv2.imwrite("closing.png", closing)