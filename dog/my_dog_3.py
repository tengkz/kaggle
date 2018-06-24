#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 21:34:57 2018

@author: ubuntu
"""

from os.path import join
img_path = join('/home/ubuntu/test/dog','menghuan1.jpeg')

import numpy as np
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing.image import load_img,img_to_array

image_size = 224

def read_and_prep_image(img_path):
    img = load_img(img_path,target_size=(image_size,image_size))
    img_array = np.array(img_to_array(img))
    return preprocess_input(np.expand_dims(img_array.copy(),axis=0))

from keras.applications import ResNet50

my_model = ResNet50(weights='imagenet')
test_data = read_and_prep_image(img_path)
pred = my_model.predict(test_data)

from keras.applications.imagenet_utils import decode_predictions
from IPython.display import Image,display

most_likely_label = decode_predictions(pred,top=3)

display(Image(img_path))
print most_likely_label[0]