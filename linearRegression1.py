# -*- coding: utf-8 -*-
# @Author: lypto
# @Date:   2017-11-28 09:44:50
# @Last Modified by:   lypto
# @Last Modified time: 2017-11-29 07:17:00


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split#导入的进行数据拆分的函数
from sklearn.linear_model import LinearRegression#导入进行线性回归的函数
from sklearn import metrics#导入计算均方等属性的函数
from sklearn.model_selection import cross_val_predict#导入交叉验证函数
from sklearn.preprocessing import scale#导入规范化函数



def lineareg(X,y):

	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)#拆分成训练集和测试集
	linreg = LinearRegression()
	linreg.fit(X_train, y_train)#这一步是计算的核心步骤，实现梯度下降
	theta0=linreg.intercept_
	theta=linreg.coef_
	#模型拟合测试集
	y_pred = linreg.predict(X_test)#对所有测试集进行预测
	print("--------------------------------------------------------the test data are listed as follow")
	# 用scikit-learn计算MSE,平均方差
	print("MSE:",metrics.mean_squared_error(y_test, y_pred))
	# 用scikit-learn计算RMSE，平方差开根号
	print ("RMSE:",np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
	predicted = cross_val_predict(linreg, X, y, cv=10)#十折交叉检验
	#计算交叉检验的MSE以及RMSE
	print("----------------------------------------------------------the follow are error analyse")
	print ("MSE:",metrics.mean_squared_error(y, predicted))
	print ("RMSE:",np.sqrt(metrics.mean_squared_error(y, predicted)))
	#绘出误差图像
	fig, ax = plt.subplots()
	ax.scatter(y, predicted, edgecolors=(0, 0, 0),label="predicted error")
	ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
	ax.set_xlabel('Measured')
	ax.set_ylabel('Predicted')
	plt.show()
	theta=np.append(theta0,theta)
	return theta

if __name__=="__main__":

	#载入数据，header规定第一列作为列名，gbk编码可以避免文件中文编码问题
	#载入的数据中包括五列，分别为AT V AP RH PE
	data=pd.read_csv("./Folds5x2_pp.csv",header=0,encoding="gbk")
	X = data[['AT', 'V', 'AP', 'RH']]
	y = data[['PE']]
	#选取AT和V两个变量进行可视化
	fig, ax = plt.subplots()
	ax.scatter(X['AT'], X['V'],c='g',label='example',marker=r'$\clubsuit$')
	ax.set_xlabel('feature:AT')
	ax.set_ylabel('feature:V')
	plt.show()
	theta=lineareg(X,y)
	print(theta)
