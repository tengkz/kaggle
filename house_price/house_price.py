# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 10:02:15 2017

@author: hztengkezhen
"""

import os
os.chdir('E:\\machine_learning\\kaggle\\house_price')

import numpy as np
import pandas as pd
import seaborn as sns

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
