# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 15:33:07 2020

@author: THILANKA
"""
import random
from tensorflow import keras
import numpy as np
import os
import cv2

from skimage.io import imread,imshow
from skimage.transform import resize
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

#Define constants
seed = 42
np.random.seed = seed
IMAGE_HEIGHT=300
IMAGE_WIDTH=300
in_channels=3


#User paths
TRAIN_PATH = ''
Train_images = os.listdir(TRAIN_PATH)
LABEL_PATH = ''
Label_images = os.listdir(LABEL_PATH)
TEST_PATH = ''
Test_images = os.listdir(TEST_PATH)
TEST_LABEL_PATH = ''
Test_Label_images = os.listdir(TEST_LABEL_PATH)
#############################################   Reading training Data   ################################

#Create empty arrays for x_train, y_train and X_test
X_train = np.zeros((len(Train_images), IMAGE_HEIGHT, IMAGE_WIDTH,in_channels), dtype=np.uint8)
Y_train = np.zeros((len(Label_images), IMAGE_HEIGHT, IMAGE_WIDTH,1), dtype=np.bool)


X_test = np.zeros((len(Test_images), IMAGE_HEIGHT, IMAGE_WIDTH,in_channels), dtype=np.uint8)
Y_test = np.zeros((len(Test_Label_images), IMAGE_HEIGHT, IMAGE_WIDTH,1), dtype=np.bool)

#Importing training data sets and respective labels
for n, image_name in enumerate(Train_images): #Remember enumerate method adds a counter and returns the enumerate object
    if (image_name.split('.')[1] == 'png'):
        img = cv2.imread(TRAIN_PATH +'/'+ image_name)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #img = np.expand_dims(resize(img, (IMAGE_HEIGHT, IMAGE_WIDTH), mode='constant', preserve_range=True), axis=-1)
        #img = resize(img, (IMAGE_HEIGHT, IMAGE_WIDTH), mode='constant', preserve_range=True)
        X_train[n] = img       
        
        label = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH,1), dtype=np.bool)
        label= imread(LABEL_PATH +'/'+ image_name)
        #label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        #label = resize(label, (IMAGE_HEIGHT, IMAGE_WIDTH), mode='constant', preserve_range=True) 
        label = np.expand_dims(resize(label, (IMAGE_HEIGHT, IMAGE_WIDTH), mode='constant', preserve_range=True), axis=-1)
        Y_train[n] = label


def accuracy_cal (prediction,test):
    prediction = prediction.flatten()
    test = test.flatten()
    print ('')
    # accuracy: (tp + tn) / (p + n)
    accuracy = accuracy_score(test, prediction)
    print('Accuracy: %f' % accuracy)
    # precision tp / (tp + fp)
    precision = precision_score(test, prediction)
    print('Precision: %f' % precision)
    # recall: tp / (tp + fn)
    recall = recall_score(test, prediction)
    print('Recall: %f' % recall)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(test, prediction)
    print('F1 score: %f' % f1)


'''
#Visualize an image and its respective labelled image to Check the code        
image_x = random.randint(0, len(Train_images))
k=X_train[image_x]
imshow(k)
plt.show()
imshow(np.squeeze(Y_train[image_x]))
plt.show()
'''

def segnet(n_levels, initial_features=16, n_blocks=2, kernel_size=3, pooling_size=2, in_channels=3, out_channels=1):
    inputs = keras.layers.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, in_channels))
    x = inputs
    
    convpars = dict(kernel_size=kernel_size, activation='relu', padding='same')
    
    #downstream
    skips = {}
    for level in range(n_levels):
        for _ in range(n_blocks):
            x = keras.layers.Conv2D(initial_features * 2 ** level, **convpars)(x)
        if level < n_levels - 1:
            skips[level] = x
            x = keras.layers.MaxPool2D(pooling_size)(x)
            
    # upstream
    for level in reversed(range(n_levels-1)):
        x = keras.layers.Conv2DTranspose(initial_features * 2 ** level, strides=pooling_size, **convpars)(x)
        for _ in range(n_blocks):
            x = keras.layers.Conv2D(initial_features * 2 ** level, **convpars)(x)
            
    # output
    activation = 'sigmoid' if out_channels == 1 else 'softmax'
    x = keras.layers.Conv2D(out_channels, kernel_size=1, activation=activation, padding='same')(x)
    
    return keras.Model(inputs=[inputs], outputs=[x], name=f'SegNet-L{n_levels}-F{initial_features}')

#############################################   Call the model with parameters  #######################################
model = segnet(3)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

#Check points
checkpointer = keras.callbacks.ModelCheckpoint('model_for_roads.h5', verbose=1, save_best_only=True)
callbacks = [
        keras.callbacks.EarlyStopping(patience=3, monitor='val_loss'),
        keras.callbacks.TensorBoard(log_dir='logs')]

#Train the model
results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=16, epochs=25, callbacks=callbacks)

#save the model
model.save(f'SegNet-RoadExtraction_{IMAGE_HEIGHT}_{IMAGE_WIDTH}_{in_channels}.h5')


############################################# Importing testing data to check #########################      
for n, image_name in enumerate(Test_images):
    if (image_name.split('.')[1] == 'png'):
        img = imread(TEST_PATH +'/'+ image_name)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #img = np.expand_dims(resize(img, (IMAGE_HEIGHT, IMAGE_WIDTH), mode='constant', preserve_range=True), axis=-1)
        #img = resize(img, (IMAGE_HEIGHT, IMAGE_WIDTH), mode='constant', preserve_range=True)
        X_test[n] = img
        
        label = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 1), dtype=np.bool)
        label= imread(TEST_LABEL_PATH +'/'+ image_name)
        #label = resize(label, (IMAGE_HEIGHT, IMAGE_WIDTH), mode='constant', preserve_range=True) 
        label = np.expand_dims(resize(label, (IMAGE_HEIGHT, IMAGE_WIDTH), mode='constant',  
                                      preserve_range=True), axis=-1)
        Y_test[n] = label
 


############################################ Apply model to training, testing and validation dataset   #######################################

pred_test = ((model.predict(X_test))>0.2).astype(np.uint8)
pred_train = ((model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1))> 0.2).astype(np.uint8)
pred_val = ((model.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1))>0.2).astype(np.uint8)

############################################# Visualize a random image to check the model output   #######################################

'''
preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)
preds_test_t = (preds_test > 0.5).astype(np.uint8)

'''

# Perform a sanity check on some random training samples
ix = random.randint(0, len(pred_train))
l=X_train[ix]
imshow(l[:, :, :])
plt.show()
imshow(np.squeeze(Y_train[ix]))
plt.show()
imshow(np.squeeze(pred_train[ix]))
plt.show()
accuracy_cal(pred_train[ix], Y_train[ix])

# Perform a sanity check on some random validation samples
ix = random.randint(0, len(pred_val))
m=X_train[int(X_train.shape[0]*0.9):][ix]
imshow(m[:, :, :])
plt.show()
imshow(np.squeeze(Y_train[int(Y_train.shape[0]*0.9):][ix]))
plt.show()
imshow(np.squeeze(pred_val[ix]))
plt.show()
accuracy_cal(pred_val[ix], Y_train[ix])

# Perform a sanity check on some random testing samples
ix = random.randint(0, len(pred_test))
n=X_test[ix]
imshow(n[:, :, :])
plt.show()
imshow(np.squeeze(Y_test[ix]))
plt.show()
imshow(np.squeeze(pred_test[ix]))
plt.show()
accuracy_cal(pred_test[ix], Y_test[ix])
