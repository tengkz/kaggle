# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 13:41:20 2017

@author: ThinkPad
"""

import os
os.chdir('E:\\machine_learning\\kaggle\\kaggle\\titanic')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

def names(train,test):
    for item in [train,test]:
        item['Name_Len'] = item['Name'].apply(lambda s:len(s))
        item['Name_Title'] = item['Name'].apply(lambda s:s.split(',')[1]).apply(lambda s:s.split('.')[0])
        del item['Name']
    return train,test

def age_impute(train,test):
    for item in [train,test]:
        item['Age_NA_Flag'] = np.where(item['Age'].isnull(),1,0)
        item_age = train.groupby(['Name_Title','Pclass'])['Age']
        item['Age'] = item_age.transform(lambda x:x.fillna(x.mean()))
    return train,test

def fam_size(train,test):
    for item in [train,test]:
        item_fs = item['SibSp']+item['Parch']
        item['Fam_Size'] = np.where(item_fs==0,'Solo',np.where(item_fs<=3,'Small','Large'))
        del item['SibSp'],item['Parch']
    return train,test

def ticket_grouped(train,test):
    for item in [train,test]:
        item['Ticket_Lett'] = item['Ticket'].apply(lambda x:str(x)[0])
        item['Ticket_Lett'] = np.where(item['Ticket_Lett'].isin(['1','2','3','S','P','C','A']),item['Ticket_Lett'],
                            np.where(item['Ticket_Lett'].isin(['4','5','6','7','8','W','L']),'Ticket_Low','Ticket_Other'))
        item['Ticket_Len'] = item['Ticket'].apply(lambda s:len(s))
        del item['Ticket']
    return train,test

def cabin(train,test):
    for item in [train,test]:
        item['Cabin_Letter'] = item['Cabin'].apply(lambda x:str(x)[0])
        item['Cabin_Num1'] = item['Cabin'].apply(lambda x:str(x).split(' ')[-1][1:])
        item['Cabin_Num1'].replace('an',np.NaN,inplace=True)
        item['Cabin_Num1'] = item['Cabin_Num1'].apply(lambda x:int(x) if not pd.isnull(x) and x!='' else np.NaN)
        item['Cabin_Num'] = pd.qcut(item['Cabin_Num1'],3)
    train = pd.concat([train,pd.get_dummies(train['Cabin_Num'],prefix='Cabin_num')],axis=1)
    test = pd.concat([test,pd.get_dummies(test['Cabin_Num'],prefix='Cabin_num')],axis=1)
    del train['Cabin_Num'],test['Cabin_Num']
    del train['Cabin_Num1'],test['Cabin_Num1']
    del train['Cabin'],test['Cabin']
    return train,test

def embarked_impute(train,test):
    for item in [train,test]:
        item['Embarked'] = item['Embarked'].fillna('S')
    return train,test

def dummies(train,test,columns=['Pclass','Embarked','Sex','Ticket_Lett','Cabin_Letter','Name_Title','Fam_Size']):
    for col in columns:
        train[col] = train[col].apply(lambda x:str(x))
        test[col] = test[col].apply(lambda x:str(x))
        good_cols = [col+'_'+i for i in train[col].unique() if i in test[col].unique()]
        train = pd.concat([train,pd.get_dummies(train[col],prefix=col)[good_cols]],axis=1)
        test = pd.concat([test,pd.get_dummies(test[col],prefix=col)[good_cols]],axis=1)
        del train[col],test[col]
    return train,test

train,test = names(train,test)
train,test = age_impute(train,test)
train,test = fam_size(train,test)
train,test = ticket_grouped(train,test)
train,test = cabin(train,test)
train,test = embarked_impute(train,test)
test['Fare'].fillna(train['Fare'].mean(),inplace=True)
train,test = dummies(train,test)
test_passengerid = test['PassengerId']
del train['PassengerId'],test['PassengerId']

from sklearn.ensemble import RandomForestClassifier
#rf = RandomForestClassifier(criterion='gini',n_estimators=700,min_samples_split=10,
                            #min_samples_leaf=1,max_features='auto',oob_score=True,
                            #random_state=1,n_jobs=-1)
rf = RandomForestClassifier(criterion='entropy',n_estimators=400,min_samples_split=4,
                            min_samples_leaf=1,max_features='auto',oob_score=True,
                            random_state=1,n_jobs=-1)
rf.fit(train.iloc[:,1:],train.iloc[:,0])
print '%.4f' % rf.oob_score_

print pd.concat((pd.DataFrame(train.iloc[:,1:].columns,columns=['Variables']),
           pd.DataFrame(rf.feature_importances_,columns=['Importance'])),axis=1).sort_values(by='Importance',ascending=False)[:20]

predictions = rf.predict(test)
passengerid = test_passengerid
test = pd.concat((pd.DataFrame(passengerid,columns=['PassengerId']),pd.DataFrame(predictions,columns=['Survived'])),axis=1)
#test.to_csv('result.csv',index=None)

#for grid search
#from sklearn.grid_search import GridSearchCV
#rf = RandomForestClassifier(max_features='auto', oob_score=True, random_state=1, n_jobs=-1)
#param_grid = { "criterion" : ["gini", "entropy"], "min_samples_leaf" : [1, 5, 10], "min_samples_split" : [2, 4, 10, 12, 16], "n_estimators": [50, 100, 400, 700, 1000]}
#gs = GridSearchCV(estimator=rf, param_grid=param_grid, scoring='accuracy', cv=3, n_jobs=-1)
#gs = gs.fit(train.iloc[:, 1:], train.iloc[:, 0])