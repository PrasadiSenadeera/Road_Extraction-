# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 16:01:30 2020
Create textures and edge detection bands to use as inputs to the Random forest and SVM classifiers

@author: THILANKA
"""

import cv2
from skimage.transform import rescale
from skimage import img_as_ubyte
import os
from matplotlib import pyplot as plt
import numpy as np

#read images
image_directory = ''
images = os.listdir(image_directory)
output_directory = ''


for i, image_name in enumerate(images): #Remember enumerate method adds a counter and returns the enumerate object
    if (image_name.split('.')[1] == 'png'):
        img = cv2.imread(image_directory +'/'+ image_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #img = rescale(img, 0.25, anti_aliasing=False)
        img= img_as_ubyte(img)
        
        #CANNY EDGE
        canny_edge = cv2.Canny(img, 0,255)   #Image, min and max values
        plt.imsave(output_directory +'/edge/'+ image_name, canny_edge, cmap ='hot')
        
        from skimage.filters import  sobel, prewitt,scharr
        
        #SOBEL
        edge_sobel = sobel(img)
        plt.imsave(output_directory +'/sobel/'+ image_name, edge_sobel, cmap ='binary')
                
        
        #PREWITT
        edge_prewitt = prewitt(img)
        plt.imsave(output_directory +'/prewitt/'+ image_name, edge_prewitt, cmap ='gist_gray')
        
        #scharr
        edge_schar=scharr(img)
        plt.imsave(output_directory +'/schar/'+ image_name,edge_schar, cmap ='gray')
        
        #GAUSSIAN with sigma=3
        from scipy import ndimage as nd
        gaussian3 = nd.gaussian_filter(img, sigma=3)
        plt.imsave(output_directory +'/gaussian3/'+ image_name, gaussian3,cmap ='gist_gray')
        
        #GAUSSIAN with sigma=5
        gaussian5 = nd.gaussian_filter(img, sigma=5)
        plt.imsave(output_directory +'/gaussian5/'+ image_name, gaussian5,cmap ='gist_gray')
        
        #Median
        Median=nd.median_filter(img, size=10)
        plt.imsave(output_directory +'/median/'+ image_name,Median, cmap ='gray')
        
        #STD
        std = nd.generic_filter(img, np.std, size=3)
        plt.imsave(output_directory +'/std/'+ image_name, std, cmap ='gray')
        
        
        
        
        
        
        
        

     
        
        
        
        
