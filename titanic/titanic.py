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
    imp = Imputer(missing_values='NaN',strategy='median',axis=0)
    x = imp.fit_transform(x)
    #x.loc[pd.isnull(x['Age']),'Age']=0
    #x.loc[pd.isnull(x['Embarked']),'Embarked']=0
    #x.loc[pd.isnull(x['Fare']),'Fare']=0
    from sklearn.preprocessing import OneHotEncoder
    enc = OneHotEncoder()
    x_1 = enc.fit_transform(x[:,[0,1,6]])
    x_2 = x[:,[2,3,4,5]]
    x = np.hstack([x_1.todense(),x_2])
    return x

train = pd.read_csv('train.csv',sep=',',header=0)
x = train[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]
y = train['Survived']
x = preprocess_data(x)

test = pd.read_csv('test.csv',sep=',',header=0)
test_x = test[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]
test_x = preprocess_data(test_x)

mode = 'test'
method = 'random_forest'

if mode == 'test':
    ratio = 0.1
    (total_num,dimension) = x.shape
    train_num = int(total_num*(1-ratio))
    test_num = total_num-train_num
    my_train_x = x[0:train_num,:]
    my_train_y = y[0:train_num]
    my_test_x = x[train_num:total_num,:]
    my_test_y = y[train_num:total_num]
    
    #train a model
    if method == 'decision_tree':
        from sklearn import tree
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(np.array(my_train_x),np.array(my_train_y))
        my_predict_y = clf.predict(np.array(my_test_x))
    elif method == 'logistic_regression':
        from sklearn import linear_model
        lgm = linear_model.LogisticRegression()
        lgm.fit(np.array(my_train_x),np.array(my_train_y))
        my_predict_y = lgm.predict(my_test_x)
    elif method == 'svm':
        from sklearn import svm
        clf = svm.SVC()
        clf.fit(np.array(my_train_x),np.array(my_train_y))
        my_predict_y = clf.predict(np.array(my_test_x))
    elif method == 'random_forest':
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=20)
        clf.fit(np.array(my_train_x),np.array(my_train_y))
        my_predict_y = clf.predict(np.array(my_test_x))
    
    precision = sum(my_predict_y==my_test_y)*1.0/len(my_test_y)
    print precision

elif mode == 'predict':
    if method == 'decision_tree':
        from sklearn import tree
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(np.array(x),np.array(y))
        test_y = clf.predict(np.array(test_x))
    elif method == 'logistic_regression':
        from sklearn import linear_model
        clf = linear_model.LogisticRegression()
        clf.fit(np.array(x),np.array(y))
        test_y = clf.predict(test_x)
    elif method == 'svm':
        from sklearn import svm
        clf = svm.SVC()
        clf.fit(np.array(x),np.array(y))
        test_y = clf.predict(test_x)
    elif method == 'random_forest':
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=20)
        clf.fit(np.array(x),np.array(y))
        test_y = clf.predict(test_x)

    test_passengers = test['PassengerId']
    ret = pd.DataFrame({"PassengerId":test_passengers,"Survived":test_y})
    ret.to_csv('result.csv',index=False)