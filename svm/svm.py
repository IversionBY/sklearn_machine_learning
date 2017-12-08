# -*- coding: utf-8 -*-
# @Author: IversionBY
# @Date:   2017-12-08 11:47:44
# @Last Modified by:   IversionBY
# @Last Modified time: 2017-12-08 15:57:57


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVR,LinearSVC
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
#beause of the previous learning,in these codes,I will not use the normalizatin and result analysing

#try to compare the SVR with the linearRegression on a same dataset

data=pd.read_csv("./Folds5x2_pp.csv",header=0,encoding="gbk")
X = data[['AT', 'V', 'AP', 'RH']]
y = data[['PE']]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10)#拆分成训练集和测试集
svr_Linear = LinearSVR(random_state=0)
svr_Linear.fit (X_train,y_train)
print("SVR_score：",svr_Linear.score(X_train,y_train))
liner=LinearRegression()
liner.fit(X_train,y_train)
print("Linearmodel_score:",liner.score(X_train,y_train))
#by doing so,in this example,you will see that linerRegresion fit better

#try to compare the svc with logisticregression on a same dataset
URL='https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'
wine_dataset=pd.read_csv(URL,header=None)
wine_dataset.columns=['class label','F1','F2','F3','F4','F5','F6','F7','F8','F9','F10','F11','F12','F13']
X,y=wine_dataset.iloc[:,1:].values,wine_dataset.iloc[:,0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10)

#Logistic
LR1=LogisticRegression(penalty='l1',C=5)#we get the parameter c in logisticregression,now we just use it
LR1.fit(X_train,y_train)
print("Logistic_score:",LR1.score(X_train,y_train))
#SVC
SVC = LinearSVC(random_state=0)
SVC.fit(X, y)
print("SVC_score:",SVC.score(X_train,y_train))
z=decision_function()
#the same result you can see on this example

#sklearn.svm.NuSVC,sklearn.svm.NuSVR could offer kernel