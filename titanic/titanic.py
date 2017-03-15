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
    from sklearn.preprocessing import Imputer
    imp = Imputer(missing_values='NaN',strategy='mean',axis=0)
    imp.fit(x)
    x = imp.transform(x)
    #x.loc[pd.isnull(x['Age']),'Age']=0
    #x.loc[pd.isnull(x['Embarked']),'Embarked']=0
    #x.loc[pd.isnull(x['Fare']),'Fare']=0
    return x


train = pd.read_csv('train.csv',sep=',',header=0)
x = train[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]
y = train['Survived']

x = preprocess_data(x)

test = pd.read_csv('test.csv',sep=',',header=0)
test_x = test[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]
test_x = preprocess_data(test_x)

from sklearn import tree

clf = tree.DecisionTreeClassifier()
clf = clf.fit(np.array(x),np.array(y))
test_y = clf.predict(np.array(test_x))

test_passengers = test['PassengerId']

ret = pd.DataFrame({"PassengerId":test_passengers,"Survived":test_y})
#ret.to_csv('result.csv',index=False)