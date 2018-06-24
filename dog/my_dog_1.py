#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 23:03:29 2018

@author: ubuntu
"""

import numpy as np
import pandas as pd
from os.path import join,exists,expanduser
from tqdm import tqdm
from sklearn.metrics import log_loss,accuracy_score
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications import xception
from keras.applications.vgg16 import preprocess_input,decode_predictions
from sklearn.linear_model import LogisticRegression

INPUT_SIZE = 299
NUM_CLASSES = 120
POOLING = 'avg'
SEED = 0
data_dir = '/data/kaggle/dog/'
labels = pd.read_csv(join(data_dir,'labels.csv'))
unique_breeds = sorted(labels['breed'].unique().tolist())
sample_submission = pd.read_csv(join(data_dir,'sample_submission.csv'))
labels['target'] = 1
labels_pivot = labels.pivot('id','breed','target').reset_index().fillna(0)
y_train = labels_pivot[unique_breeds].values

def read_img(img_id,train_or_test,size):
    img = image.load_img(join(data_dir,train_or_test,'%s.jpg' % img_id),target_size=size)
    img = image.img_to_array(img)
    return img

x_train = np.zeros((len(labels),INPUT_SIZE,INPUT_SIZE,3),dtype='float32')
for i,img_id in tqdm(enumerate(labels['id'])):
    img = read_img(img_id,'train',(INPUT_SIZE,INPUT_SIZE))
    x = xception.preprocess_input(np.expand_dims(img.copy(),axis=0))
    x_train[i] = x
print 'Train Image shape: {} size: {:,}'.format(x_train.shape,x_train.size)

x_test = np.zeros((len(sample_submission),INPUT_SIZE,INPUT_SIZE,3),dtype='float32')
for i,img_id in tqdm(enumerate(sample_submission['id'])):
    img = read_img(img_id,'test',(INPUT_SIZE,INPUT_SIZE))
    x = xception.preprocess_input(np.expand_dims(img.copy(),axis=0))
    x_test[i] = x
print 'Test Image shape: {} size: {:,}'.format(x_test.shape,x_test.size)

print x_train.shape
xception_bottleneck = xception.Xception(weights='imagenet',include_top=False,pooling=POOLING)
train_x_bf = xception_bottleneck.predict(x_train,batch_size=32,verbose=1)
test_x_bf = xception_bottleneck.predict(x_test,batch_size=32,verbose=1)
print 'Xception train bottleneck features shape: {} size: {:,}'.format(train_x_bf.shape,train_x_bf.size)

logreg = LogisticRegression(multi_class='multinomial',solver='lbfgs',random_state=SEED)
logreg.fit(train_x_bf,(y_train*range(NUM_CLASSES)).sum(axis=1))
train_probs = logreg.predict_proba(train_x_bf)
train_preds = logreg.predict(train_x_bf)
print 'Xception train loss: {}'.format(log_loss(y_train,train_probs))
print 'Xception train accuracy: {}'.format(accuracy_score((y_train*range(NUM_CLASSES)).sum(axis=1),train_preds))

test_probs = logreg.predict_proba(test_x_bf)
test_preds = logreg.predict(test_x_bf)

result = pd.DataFrame(data=test_probs,index=sample_submission.id,columns=unique_breeds,dtype='float32',copy=True)
result.to_csv('my_submission.csv')