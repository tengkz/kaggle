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

linear_features = ['LotFrontage','LotArea','YearBuilt','YearRemodAdd',
                  'MasVnrArea','BsmtFinSF1','BsmtUnfSF','BsmtFinSF2','TotalBsmtSF',
                  '1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea',
                  'BsmtFullBath','FullBath','HalfBath','BedroomAbvGr',
                  'KitchenAbvGr','TotRmsAbvGrd','Fireplaces','GarageYrBlt',
                  'GarageCars','GarageArea','WoodDeckSF','OpenPorchSF',
                  'EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea',
                  'MiscVal','MoSold','YrSold']
cat_features = []
for feature in train.columns:
    if feature not in linear_features and feature not in ('SalePrice','Id',
                                                          'Alley','FireplaceQu',
                                                          'PoolQC','Fence',
                                                          'MiscFeature'):
        cat_features.append(feature)

data = pd.concat([train,test],axis=0)
del data['Alley'],data['FireplaceQu'],data['PoolQC'],data['Fence'],data['MiscFeature']
num_instance,num_features = data.shape

for feature in linear_features:
    na_num = data[feature].isnull().sum()
    if na_num == 0:
        continue
    elif na_num>=num_instance/2:
        data[feature].fillna(0,inplace=True)
    else:
        data[feature].fillna(data[feature].mean(),inplace=True)

for feature in cat_features:
    na_num = data[feature].isnull().sum()
    if na_num == 0:
        continue
    elif na_num>=num_instance/2:
        data[feature].fillna('null',inplace=True)
    else:
        data[feature].fillna(data[feature].value_counts().idxmax(),inplace=True)

def process_linear_data(data):
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
    for feature in linear_features:
        data[feature] = scaler.fit_transform(data[feature])
    return data

def process_cat_data(data):
    for feature in cat_features:
        dummy_feature = pd.get_dummies(data[feature],prefix=feature)
        data = pd.concat([data,dummy_feature],axis=1)
    return data

def post_process_cat_data(data):
    for feature in cat_features:
        del data[feature]
    return data
    
#data = process_linear_data(data)
data = process_cat_data(data)
data = post_process_cat_data(data)

all_x = data.loc[:,(data.columns!='SalePrice') & (data.columns!='Id')]
all_y = data.loc[:,data.columns=='SalePrice']

train_valid_x = all_x.iloc[0:1460,:]
test_x = all_x.iloc[1460:2919,:]
train_valid_y = all_y.iloc[0:1460]
test_y = all_y.iloc[1460:2919]

from sklearn.cross_validation import train_test_split
train_x,valid_x,train_y,valid_y = train_test_split(train_valid_x,train_valid_y,
                                                   test_size=0.2,random_state=0)

#from sklearn.ensemble import RandomForestRegressor
#model = RandomForestRegressor(n_estimators=100, max_features='auto',
#                              min_samples_split=5,min_samples_leaf=2,
#                              n_jobs=-1,random_state=1,oob_score=True)

from sklearn.linear_model import Ridge
model = Ridge(alpha = 100., fit_intercept=True,normalize=True)
#from sklearn.linear_model import LassoLars
#model = LassoLars(alpha=200)

model.fit(train_x,train_y)
print model.score(valid_x,valid_y)
predict_y = model.predict(valid_x)
predict_train_y = model.predict(train_x)
predict_y[predict_y<=0]=0.001
from math import log10,sqrt
log_mse = ((np.array(map(log10,predict_y))-np.array(map(log10,valid_y.values)))**2).mean()
log_mse_train = ((np.array(map(log10,predict_train_y))-np.array(map(log10,train_y.values)))**2).mean()
print sqrt(log_mse_train),sqrt(log_mse)

test_y = model.predict(test_x)
ret = pd.DataFrame()
ret['Id'] = test['Id']
ret['SalePrice'] = test_y
ret.to_csv('result.csv',index=False)

#from sklearn.grid_search import GridSearchCV
#model = RandomForestRegressor(max_features='auto',n_jobs=-1,random_state=0,
#                              oob_score=True)
#param_grid = {'n_estimators':[50,100,500],'min_samples_split':[5,10,20,40],
#              'min_samples_leaf':[2,5,10,20]}
#gs = GridSearchCV(estimator=model,cv=5,param_grid=param_grid,scoring='mean_squared_error')
#gs.fit(train_valid_x,train_valid_y)