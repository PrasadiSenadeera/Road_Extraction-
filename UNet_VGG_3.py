
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 14:01:58 2020

@author: THILANKA
"""
import random
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import cv2

from skimage.io import imread,imshow
from skimage.transform import resize
import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Sequential, Model
from scipy import ndimage
from skimage.morphology import skeletonize
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


seed = 42
np.random.seed = seed
im_height=256
im_width=256
im_channels=3


def read_images (images_path,label_path):
    list_images = os.listdir(images_path)
    list_labels = os.listdir(label_path)
    
    #Create empty arrays for x_train, y_train and X_test
    images = np.zeros((len(list_images), im_height, im_width,im_channels), dtype=np.uint8)
    labels = np.zeros((len(list_labels), im_height, im_width,1), dtype=np.bool)
    
     #Importing training data sets and respective labels
    for n, image_name in enumerate(list_images): 
        if (image_name.split('.')[1] == 'png'):
            img = cv2.imread(images_path +'/'+ image_name)
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            #img = np.expand_dims(resize(img, (im_height, im_width), mode='constant', preserve_range=True), axis=-1)
            img = resize(img, (im_height, im_width), mode='constant', preserve_range=True)
            images[n] = img       
            
            label = np.zeros((im_height, im_width), dtype=np.bool)
            label= cv2.imread(label_path +'/'+ image_name)
            #print (label.shape)
            label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
            label = resize(label, (im_height, im_width), mode='constant', preserve_range=True) 
            label = np.expand_dims(resize(label, (im_height, im_width), mode='constant', preserve_range=True), axis=-1)
            labels[n] = label
    return images, labels

'''
X_train,Y_train = read_images('road_segmentation/training/input_300-t','road_segmentation/training/output_300-t')
#X_test,Y_test = read_images('data/Training/Images_1','data/Training/Labels_1')
image_x = random.randint(0, len(X_train))
k=X_train[image_x]
imshow(k[:,:])
plt.show()
l=Y_train[image_x]
imshow(l[:,:,0])
plt.show()
 '''  
def plot_hist(hist):    
    acc = results.history['accuracy']
    val_acc = results.history['val_accuracy']
    loss = results.history['loss']
    val_loss = results.history['val_loss']
    epochs = range(1, len(acc)+1)
    
    plt.plot(epochs, acc, 'g', label='Training accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()    
    plt.figure()
    
    plt.plot(epochs, loss, 'g', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()    
    plt.show()

def visualize(images, labels, prediction):
    #ix=5
    ix = random.randint(0, (len(prediction)-1))
    n=images[ix]
    imshow(n[:, :, :])
    plt.show()
    imshow(np.squeeze(labels[ix]))
    plt.show()
    imshow(np.squeeze(prediction[ix]))
    plt.show()
    
    final = post_processing((prediction[ix]),3)
    accuracy_cal (prediction[ix],labels[ix])    
    return final 

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
   

def post_processing(image,x):
    print (len(image))
    if (len(image))==256:
        out_image = np.zeros((len(image),im_height, im_width), dtype=np.bool)
        mor_test = image
        mor_test = np.squeeze(mor_test, axis=2)
        mor_test = ndimage.binary_closing(mor_test, structure=np.ones((x,x))).astype(int)
        mor_test = ndimage.binary_opening(mor_test, structure=np.ones((x,x))).astype(int)
        out_image = mor_test
        
        plt.show()
        n=out_image
        imshow(n)
        plt.show()
        
    else:
        out_image = np.zeros((len(image),im_height, im_width), dtype=np.bool)
        for i in range (((len(image))-1)):
            mor_test= image[i,:,:]
            mor_test = np.squeeze(mor_test, axis=2)
            mor_test = ndimage.binary_closing(mor_test, structure=np.ones((x,x))).astype(int)
            mor_test = ndimage.binary_opening(mor_test, structure=np.ones((x,x))).astype(int)
            #skeleton = skeletonize(mor_test)
            out_image[i]= mor_test
        
        ix = random.randint(0, (len(image)-1))
            
        m=np.squeeze(image[ix])
        imshow(m)
        plt.show()
        
        n=out_image[ix]
        imshow(n)
        plt.show()
    
    return out_image

#VGG model
VGG_model = VGG16(weights='imagenet', include_top=False, input_shape=(im_height, im_width,im_channels))
for layer in VGG_model.layers:
	layer.trainable = False
VGG_model.summary()

#New model only from a part of VGG
#new_model = Model(inputs=VGG_model.input, outputs=VGG_model.get_layer('block4_conv3').output)
#new_model.summary()

inputs = inputs=VGG_model.input


c1 = VGG_model.get_layer('block1_conv2').output
p1 = VGG_model.get_layer('block1_pool').output

c2 = VGG_model.get_layer('block2_conv1').output
p2 = VGG_model.get_layer('block2_pool').output


c3 = VGG_model.get_layer('block3_conv3').output



#Expansive path 
u4 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c3)
u4 = tf.keras.layers.concatenate([u4, c2])
c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u4)
c4 = tf.keras.layers.Dropout(0.2)(c4)
c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
 
u5 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c4)
u5 = tf.keras.layers.concatenate([u5, c1])
c5 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u5)
c5 = tf.keras.layers.Dropout(0.2)(c5)
c5 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
 

outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c5)
model = tf.keras.Model(inputs=[inputs], outputs=[outputs]) #model is developed from input to output

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) #optimizer try to minimize the loss function
model.summary()
checkpointer = keras.callbacks.ModelCheckpoint('model_for_roads.h5', verbose=1, save_best_only=True)
callbacks = [
        keras.callbacks.EarlyStopping(patience=3, monitor='val_loss'),
        keras.callbacks.TensorBoard(log_dir='logs')]


X_train,Y_train = read_images('road_segmentation/training/input_300_1','road_segmentation/training/output_300_1')
#X_train,Y_train = read_images('data/Training/Images_1','data/Training/Labels_1')

features=VGG_model.predict(X_train)

results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=4, epochs=10, callbacks=callbacks)
model.save(f'UNET-RoadSegmentation_levels_3_{im_height}_{im_channels}.h5')

# Display accuracy as a graph
plot_hist(results)

#import training data
X_test,Y_test = read_images('road_segmentation/testing/input_300','road_segmentation/testing/output_300')
#X_test, Y_test = read_images('data/Training/Images_2','data/Training/Labels_2')


############################################ Apply model to training, testing and validation dataset   #######################################

predict_test = model.predict(X_test, verbose=1)
predict_train = model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)
preddict_val = model.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1)

pred_test= (predict_test>0.3).astype(np.bool)
pred_train= (predict_train>0.3).astype(np.uint8)
pred_val= (preddict_val >0.3).astype(np.uint8)

############################################# Visualize a random image (training, validation and Testing) to check the model output   ###########################################


print ("Prediction on training data")
Final_Train = visualize(X_train,Y_train,pred_train)


print ("Prediction on validation data")
#ix=5
ix = random.randint(0, (len(pred_val)-1))
imshow(X_train[int(X_train.shape[0]*0.9):][ix])
plt.show()
imshow(np.squeeze(Y_train[int(Y_train.shape[0]*0.9):][ix]))
plt.show()
imshow(np.squeeze(pred_val[ix]))
plt.show()
Final_val= post_processing(pred_val[ix],3)
accuracy_cal(pred_val[ix], Y_train[ix])


print ("Prediction on testing data")
Final_test = visualize(X_test,Y_test,pred_test)

######################################### Post processing ###############################
