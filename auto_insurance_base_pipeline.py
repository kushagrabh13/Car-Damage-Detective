import numpy as np
import cv2
import keras
import tensorflow as tf
import os
import matplotlib.pyplot as plt

from keras.layers.convolutional import Conv2D, Conv2DTranspose, UpSampling2D, ZeroPadding2D
from keras.layers.pooling import MaxPool2D, AvgPool2D, GlobalAveragePooling2D
from keras.layers import Dense, Flatten, Input, BatchNormalization, Dropout, Reshape, Lambda, Activation
from keras.models import Model, Sequential
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import keras.backend as K
from keras.applications.vgg16 import VGG16
from keras.initializers import RandomNormal
from keras.optimizers import Adam
from keras import regularizers
from keras import initializers
from keras.layers.merge import concatenate
from keras.regularizers import l2
import h5py

class AutoInsurance():

    def __init__(self):
        self.weights_path = './models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
        self.input_shape = (224, 224, 3)

    def is_damaged_model(self):
        model = Sequential()
        model.add(VGG16(include_top=False, input_shape=self.input_shape, weights=self.weights_path))
        
        for layer in model.layers:
            layer.trainable=False
            
        top_model = Sequential()
        top_model.add(GlobalAveragePooling2D(input_shape=model.output_shape[1:]))
        top_model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.01)))
        top_model.add(Dropout(0.5))
        top_model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
        top_model.add(Dropout(0.5))
        top_model.add(Dense(1, activation='sigmoid'))
        
        model.add(top_model)
        model.compile(loss='binary_crossentropy',
                     optimizer = 'adam', metrics=['accuracy'])
        model.load_weights('./models/vgg16_id.h5')

        return model

    def damage_location_model(self):
        model = Sequential()
        model.add(VGG16(include_top=False, input_shape=self.input_shape, weights=self.weights_path))
        
        for layer in model.layers:
            layer.trainable=False
        
        top_model = Sequential()
        top_model.add(GlobalAveragePooling2D(input_shape=model.output_shape[1:]))
        top_model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.01)))
        top_model.add(Dropout(0.5))
        top_model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
        top_model.add(Dropout(0.5))
        top_model.add(Dense(3, activation='softmax'))
        
        model.add(top_model)
        model.compile(loss='categorical_crossentropy',
                     optimizer = 'adam', metrics=['accuracy'])    
        
        model.load_weights('./models/vgg16_dl_2.h5')

        return model
    
    def damage_severity_model(self):
        model = Sequential()
        model.add(VGG16(include_top=False, input_shape=self.input_shape, weights=self.weights_path))
        
        for layer in model.layers:
            layer.trainable=False
        
        top_model = Sequential()
        top_model.add(GlobalAveragePooling2D(input_shape=model.output_shape[1:]))
        top_model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.01)))
        top_model.add(Dropout(0.5))
        top_model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
        top_model.add(Dropout(0.5))
        top_model.add(Dense(3, activation='softmax'))
        
        model.add(top_model)
        model.compile(loss='categorical_crossentropy',
                     optimizer = 'adam', metrics=['accuracy'])    
        
        model.load_weights('./models/vgg16_s.h5')
        
        return model
    
    def preprocess_image(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.resize(image, (224,224))
        image = np.expand_dims(image, axis=0)
        return image

    def predict(self, image_path):
        image = self.preprocess_image(image_path)
        is_damaged_flag = False
        damage_location = ''
        damage_severity = ''
        damaged_prob = self.is_damaged_model().predict(image)[0][0]
        if damaged_prob >= 0.5:
            is_damaged_flag = True

        if is_damaged_flag:
            K.clear_session()
            damage_location = self.damage_location_model().predict(image)[0]
            damage_location = np.argmax(damage_location)
            
            if damage_location == 0:
                damage_location = 'Front'
            
            elif damage_location == 1:
                damage_location = 'Rear'
            
            else:
                damage_location = 'Side'

            K.clear_session()
            damage_severity = self.damage_severity_model().predict(image)[0]
            damage_severity = np.argmax(damage_severity)

            if damage_severity == 0:
                damage_severity = 'Minor'
            
            elif damage_severity == 1:
                damage_severity = 'Moderate'
            
            else:
                damage_severity = 'Severe'

        else:
            print('The car is not damaged !!')

        return is_damaged_flag, damage_location, damage_severity

    
if __name__ == '__main__':
    image_path = '../2.png'
    ai = AutoInsurance()
    print(ai.predict(image_path))