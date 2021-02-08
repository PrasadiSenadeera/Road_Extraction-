# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 20:19:30 2020

@author: THILANKA
"""


import cv2
import pandas as pd
from skimage.filters import  sobel, prewitt
#from skimage.transform import rescale
import numpy as np

def features (img):
    
    df = pd.DataFrame()
    img2 = img.reshape(-1)
    df['Original Image'] = img2    
       
    #SOBEL
    edge_sobel = sobel(img)
    edge_sobel1 = edge_sobel.reshape(-1)
    df['sobel']=edge_sobel1
            
    #CANNY EDGE
    edges = cv2.Canny(img, 0,255)   #Image, min and max values
    edges1 = edges.reshape(-1)
    df['canny_edge']=edges1
    
    #PREWITT
    edge_prewitt = prewitt(img)
    edge_prewitt1 = edge_prewitt.reshape(-1)
    df['pwewitt']=edge_prewitt1
    
    
    #GAUSSIAN with sigma=3
    from scipy import ndimage as nd
    gaussian3 = nd.gaussian_filter(img, sigma=3)
    gaussian31 = gaussian3.reshape(-1)
    df['gaussian']=gaussian31
    
    #STD
    std = nd.generic_filter(img, np.std, size=3)
    std1 = std.reshape(-1)
    df['std']=std1
    
    return df

#Applying trained model to segment multiple files.
    

import pickle
from matplotlib import pyplot as plt
from skimage import img_as_ubyte

filename = "RF_model_300"
loaded_model = pickle.load(open(filename, 'rb'))

img1= cv2.imread('road_segmentation/validation/input_300/img-6_6.png')
img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
#img = rescale(img, 0.25, anti_aliasing=False) 
img= img_as_ubyte(img)


true=cv2.imread('road_segmentation/validation/output_300/img-6_6.png')
true = cv2.cvtColor(true, cv2.COLOR_BGR2GRAY)
#true = rescale(true, 0.25, anti_aliasing=False)
true= img_as_ubyte(true) 
true1 = true.reshape(-1)


X = features(img)
result = loaded_model.predict(X)
segmented = result.reshape((img.shape))


from sklearn import metrics
print ("Accuracy on testing data = ", metrics.accuracy_score(true1, result))
#plt.imsave('Results/Support vector.png', segmented, cmap ='gray')

from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(true1, result))
print(classification_report(true1, result))


plt.imshow(segmented, cmap='Greys')












