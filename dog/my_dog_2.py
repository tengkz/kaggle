#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 20:22:25 2018

@author: ubuntu
"""

from os.path import join

image_dir = '/data/kaggle/dog/train'
img_paths = [join(image_dir,filename) for filename in 
             ['0246f44bb123ce3f91c939861eb97fb7.jpg',
              '84728e78632c0910a69d33f82e62638c.jpg',
              '8825e914555803f4c67b26593c9d5aff.jpg',
              '91a5e8db15bccfb6cfa2df5e8b95ec03.jpg']]

import numpy as np
#from keras.applications.resnet50 import preprocess_input
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing.image import load_img,img_to_array

image_size = 224

def read_and_prep_images(img_paths,img_height=image_size,img_width=image_size):
    imgs = [load_img(img_path,target_size=(img_height,img_width)) for img_path in img_paths]
    img_array = np.array([img_to_array(img) for img in imgs])
    return preprocess_input(img_array)

#from tensorflow.python.keras.applications import ResNet50
from keras.applications import ResNet50

my_model = ResNet50(weights='imagenet')
test_data = read_and_prep_images(img_paths)
preds = my_model.predict(test_data)

from keras.applications.imagenet_utils import decode_predictions
from IPython.display import Image,display

most_likely_labels = decode_predictions(preds,top=3)

for i,img_path in enumerate(img_paths):
    display(Image(img_path))
    print most_likely_labels[i]