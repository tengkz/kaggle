# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 16:06:36 2017

@author: hztengkezhen
"""

import os
os.chdir('E:\\machine_learning\\kaggle\\titanic')

import numpy as np
import pandas as pd

def preprocess_data(x):
    #process categrial variables
    x.loc[x['Sex']=='male','Sex']=1
    x.loc[x['Sex']=='female','Sex']=2
    x.loc[x['Embarked']=='C','Embarked']=1
    x.loc[x['Embarked']=='Q','Embarked']=2
    x.loc[x['Embarked']=='S','Embarked']=3
    #impute nan
    from sklearn.preprocessing import Imputer
    imp = Imputer(missing_values='NaN',strategy='most_frequent',axis=0)
    x = imp.fit_transform(x)  
    #OneHotEncoder
    from sklearn.preprocessing import OneHotEncoder
    enc = OneHotEncoder()
    x_1 = enc.fit_transform(x[:,[0,1,6]])
    x_2 = x[:,[2,3,4,5]]

    x = np.hstack([x_1.todense(),x_2])
    return x

def scale_data(train_x,test_x):
    train_x_1 = train_x[:,0:8]
    train_x_2 = train_x[:,8:12]
    test_x_1 = test_x[:,0:8]
    test_x_2 = test_x[:,8:12]
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import MinMaxScaler
    standard_scaler = StandardScaler().fit(train_x_2)
    train_x_2 = standard_scaler.transform(train_x_2)
    test_x_2 = standard_scaler.transform(test_x_2)
    min_max_scaler = MinMaxScaler().fit(train_x_1)
    train_x_1 = min_max_scaler.transform(train_x_1)
    test_x_1 = min_max_scaler.transform(test_x_1)
    return np.hstack([train_x_1,train_x_2]),np.hstack([test_x_1,test_x_2])

#load data
train = pd.read_csv('train.csv',sep=',',header=0)
x = train[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]
y = train['Survived']
x = preprocess_data(x)

test = pd.read_csv('test.csv',sep=',',header=0)
test_x = test[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]
test_x = preprocess_data(test_x)

x,test_x = scale_data(x,test_x)

mode = 'predict'
method = 'svm'

#model selection
if method == 'decision_tree':
    from sklearn import tree
    clf = tree.DecisionTreeClassifier()
elif method == 'logistic_regression':
    from sklearn import linear_model
    clf = linear_model.LogisticRegression()
elif method == 'svm':
    from sklearn import svm
    clf = svm.SVC(kernel='rbf')
elif method == 'random_forest':
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=20)

if mode == 'test':
    from sklearn.cross_validation import train_test_split
    my_train_x,my_test_x,my_train_y,my_test_y = train_test_split(
        np.array(x),np.array(y),test_size=0.1,random_state=0)
    clf.fit(np.array(my_train_x),np.array(my_train_y))
    print clf.score(my_test_x,my_test_y)
elif mode == 'predict':
    clf.fit(np.array(x),np.array(y))
    test_y = clf.predict(test_x)
    test_passengers = test['PassengerId']
    ret = pd.DataFrame({"PassengerId":test_passengers,"Survived":test_y})
    ret.to_csv('result.csv',index=False)