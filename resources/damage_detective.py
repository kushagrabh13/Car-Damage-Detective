def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import numpy as np
import cv2
import keras
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import base64
from PIL import Image
import io

from flask_restful import Resource

from keras.applications.densenet import DenseNet121
from keras.layers.convolutional import Conv2D, Conv2DTranspose, UpSampling2D, ZeroPadding2D
from keras.layers.pooling import MaxPool2D, AvgPool2D, GlobalAveragePooling2D
from keras.layers import Dense, Flatten, Input, BatchNormalization, Dropout, Reshape, Lambda, Activation
from keras.models import load_model, Sequential
from keras.optimizers import Adam
import keras.backend as K
import h5py

mapping = {

     0: 'TailLight',
     1: 'Bumper',
     2: 'Hood',
     3: 'Left Fender',
     4: 'Left Headlight',
     5: 'Rear',
     6: 'Right Fender',
     7: 'Right Headlight',
     8: 'Side Window'
}

class DamageDetective(Resource):

    def __init__(self):
        self.is_damaged_model_path = './models/vgg16_id.h5'
        self.damaged_part_model_path = './models/DenseNet-BC-121-32-no-top.h5'
        self.damage_severity_model_path = './models/vgg16_s.h5'
        self.input_shape = (224, 224, 3)

    def is_damaged_model(self):
        model = load_model(self.is_damaged_model_path)
        return model

    def damaged_part_model(self):
        model = Sequential()
        model.add(DenseNet121(include_top=False, input_shape=self.input_shape, pooling='avg', weights=self.damaged_part_model_path))

        for layer in model.layers[0].layers[:350]:
            layer.trainable=False

        model.add(Dense(9, activation='softmax'))

        model.compile(loss='categorical_crossentropy',
                     optimizer = 'adam', metrics=['accuracy'])

        model.load_weights('./models/vgg16_dp_dense.h5')

        return model

    def damage_severity_model(self):
        model = load_model(self.damage_severity_model_path)
        return model
    
    @classmethod
    def preprocess_image(cls, image_path): # For Testing Only
        image = cv2.imread(image_path)
        image = cv2.resize(image, (224,224))
        image = np.expand_dims(image, axis=0)
        return image

    @classmethod
    def load_base64(cls, image): # For REST API use this instead
        tmp = base64.b64decode(image)
        img = Image.open(io.BytesIO(tmp))
        img = np.array(img)
        o_size = img.shape[:-1]
        # if len(img.shape) > 2:
        #         img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img, (224,224))
        img = np.expand_dims(img, axis=0)
        return img, o_size
    
    def predict(self, image, filename):
        image, o_size = self.load_base64(image)
        cv2.imwrite('static/uploads/'+filename, cv2.resize(cv2.cvtColor(image[0,:,:,:], cv2.COLOR_BGR2RGB), (o_size[1], o_size[0])))
        #image = self.preprocess_image(image_path)
        is_damaged_flag = False
        damage_location = ''
        damage_severity = ''
        damaged_prob = self.is_damaged_model().predict(image)[0][0]
        if damaged_prob >= 0.5:
            is_damaged_flag = True

        if is_damaged_flag:
            K.clear_session()
            # damage_location = self.damage_location_model().predict(image)[0]
            # damage_location = np.argmax(damage_location)
            
            # if damage_location == 0:
            #     damage_location = 'Front'
            
            # elif damage_location == 1:
            #     damage_location = 'Rear'
            
            # else:
            #     damage_location = 'Side'

            damaged_part = self.damaged_part_model().predict(image)[0]
            
            # for i in range(len(mapping)):
            #     print(mapping[i] + ': ', round(damaged_part[i]*100, 2))
            damaged_part_args = np.argsort(damaged_part)[-3:]
            damaged_parts = []
            
            # print(damaged_part)
            for i in damaged_part_args:
                    if damaged_part[i] >= 0.05:
                        damaged_parts.append(mapping[i])
            # damaged_part = [mapping[damaged_part[i]] for i in range(len(damaged_part))]

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
            return {'is_damaged': is_damaged_flag, 'damage_location': None, 'damage_severity': None}

        return {'is_damaged': is_damaged_flag, 'damage_location': damaged_parts, 'damage_severity': damage_severity}
  
if __name__ == '__main__':
    ai = AutoInsurance()
    images = []
    for folder in os.listdir('/mnt/d/data'):
        for image in os.listdir('/mnt/d/data/'+ folder):
            if image == '4.jpg':
                images.append('/mnt/d/data/' + folder + '/' + image)
    
    for i in range(len(images)):
        print(images[i], ai.predict(images[i]))
