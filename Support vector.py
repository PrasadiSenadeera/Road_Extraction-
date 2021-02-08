# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 20:23:13 2020

@author: THILANKA
"""


# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 16:25:37 2020

@author: THILANKA
"""

import numpy as np
import cv2
import pandas as pd
#from skimage.transform import rescale
from skimage import img_as_ubyte
from matplotlib import pyplot as plt
import os


#read images
image_directory = 'road_segmentation/validation/input_300_1'
images = os.listdir(image_directory)
label_directory = 'road_segmentation/validation/output_300_1'
L_images = os.listdir(label_directory)

#Create an empty Data frame 
df = pd.DataFrame()


#place the holders to save data
original = []
labels = []
sobel_out=[]
canny_out=[]
prewitt_out=[]
gaussian_out=[]
std_out=[]

for i, image_name in enumerate(images): #Remember enumerate method adds a counter and returns the enumerate object
    if (image_name.split('.')[1] == 'png'):
        img = cv2.imread(image_directory +'/'+ image_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #img = rescale(img, 0.25, anti_aliasing=False)
        img= img_as_ubyte(img)
        img2 = img.reshape(-1) #reshape images
        original.append(img2)            
                
        #Apply filters
        
        from skimage.filters import  sobel, prewitt
        
        #SOBEL
        edge_sobel = sobel(img)
        edge_sobel1 = edge_sobel.reshape(-1)
        sobel_out.append(np.array(edge_sobel1))
                
        #CANNY EDGE
        edges = cv2.Canny(img, 0,255)   #Image, min and max values
        edges1 = edges.reshape(-1)
        canny_out.append(np.array(edges1))
        
        #PREWITT
        edge_prewitt = prewitt(img)
        edge_prewitt1 = edge_prewitt.reshape(-1)
        prewitt_out.append(np.array(edge_prewitt1))
        
        #GAUSSIAN with sigma=3
        from scipy import ndimage as nd
        gaussian3 = nd.gaussian_filter(img, sigma=3)
        gaussian31 = gaussian3.reshape(-1)
        gaussian_out.append(np.array(gaussian31))
        
        #STD
        
        std = nd.generic_filter(img, np.std, size=3)
        std1 = std.reshape(-1)
        std_out.append(np.array(std1))       
        
        #Labelled Image
        lab_img = cv2.imread(label_directory +'/'+ image_name)
        lab_img = cv2.cvtColor(lab_img, cv2.COLOR_BGR2GRAY)
        # lab_img = rescale(lab_img, 0.25, anti_aliasing=False) 
        lab_img= img_as_ubyte(lab_img)
        lab_img2 = lab_img.reshape(-1)
        labels.append(np.array(lab_img2))




#Creae a data frame with values
      
def flat_list(list):
    f_list = [ item for elem in list for item in elem]
    return f_list

f_original = flat_list(original)
df['image'] = f_original

f_sobel = flat_list(sobel_out)
df['Sobel'] = f_sobel 
  
f_canny = flat_list(canny_out)
df['Canny Edge'] = f_canny

f_prewitt = flat_list(prewitt_out) 
df['Prewitt'] = f_prewitt

f_Gaussian= flat_list(gaussian_out)
df['Gaussian']=f_Gaussian

f_std= flat_list(std_out)
df['STD']=f_std

f_labels = flat_list(labels)
df['Labels'] = f_labels



plt.imshow(img)
plt.imshow(lab_img)


#print(len(f_original))
#print(df.head())
#print(df.shape)


########################### RANDOM FOREST CLASSIFIER ################################


#Define the dependent variable that needs to be predicted (labels)
Y = df["Labels"].values

#Define the independent variables
X = df.drop(labels = ["Labels"], axis=1) 


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=20)

from sklearn.svm import LinearSVC
# Instantiate model with n number of decision trees
model = LinearSVC(max_iter=4000)

# Train the model on training data
model.fit(X_train, y_train)

#test prediction on the training data itself. Should be good. 
prediction_test_train = model.predict(X_train)

#Test prediction on testing data. 
prediction_test = model.predict(X_test)


from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, prediction_test))
print(classification_report(y_test, prediction_test))


#Check the accuracy on test data
from sklearn import metrics

# accuracy on training data. 
print ("Accuracy on training data = ", metrics.accuracy_score(y_train, prediction_test_train))
#Check accuracy on test dataset.
print ("Accuracy on testing data = ", metrics.accuracy_score(y_test, prediction_test))


#Save model and apply for new data
import pickle

#Save the trained model as pickle string to disk for future use
filename = "SV_model_300_4000"
pickle.dump(model, open(filename, 'wb'))







