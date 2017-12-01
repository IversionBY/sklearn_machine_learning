# -*- coding: utf-8 -*-
# @Author: IversionBY
# @Date:   2017-11-28 09:44:50
# @Last Modified by:   IversionBY
# @Last Modified time: 2017-12-01 09:52:34


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split#导入的进行数据拆分的函数
from sklearn.linear_model import LinearRegression#导入进行线性回归的函数
from sklearn import metrics#导入计算均方等属性的函数
from sklearn.model_selection import cross_val_predict#导入交叉验证函数
from sklearn.preprocessing import StandardScaler#导入标准化函数
from sklearn.externals import joblib#模型持久化

def lineareg(X,y):

	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10)#拆分成训练集和测试集
	#进行参数标准化
	stdsc=StandardScaler()
	X_train_std=stdsc.fit_transform(X_train)
	X_test_std=stdsc.transform(X_test)

	#进行线性回归拟合
	linreg = LinearRegression()
	linreg.fit(X_train_std, y_train)#这一步是计算的核心步骤，实现梯度下降
	joblib.dump(linreg, "train_model_linreg.m")
	theta0=linreg.intercept_#参数获取
	theta=linreg.coef_

	#模型拟合测试集
	print("--------------------------------------------------------the test data are listed as follow")
	y_pred = linreg.predict(X_test_std)#对所有测试集进行预测
	# 用scikit-learn计算MSE,平均方差
	print("TSET_MSE:",metrics.mean_squared_error(y_test, y_pred))
	# 用scikit-learn计算RMSE，平方差开根号
	print ("TEST_RMSE:",np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
	
	#计算交叉检验的MSE以及RMSE
	print("----------------------------------------------------------the follow are cv analyse")
	predicted = cross_val_predict(linreg, X, y, cv=10)#十折交叉检验
	print ("CV_MSE:",metrics.mean_squared_error(y, predicted))
	print ("CV_RMSE:",np.sqrt(metrics.mean_squared_error(y, predicted)))

	#绘制回归直观图像
	fig, ax = plt.subplots()
	ax.scatter(y, predicted, edgecolors=(0, 0, 0))
	ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=6)
	ax.set_xlabel('Measured')
	ax.set_ylabel('Predicted')
	plt.title("predicted situation")

	plt.show()
	theta=np.append(theta0,theta)
	return theta#传回权重系数

if __name__=="__main__":
	'''
	载入数据，header规定第一列作为列名，gbk编码可以避免文件中文编码问题
	载入的数据中包括五列，分别为AT V AP RH PE

	'''
	data=pd.read_csv("./Folds5x2_pp.csv",header=0,encoding="gbk")
	X = data[['AT', 'V', 'AP', 'RH']]
	y = data[['PE']]
	
	#选取AT和V两个变量进行可视化
	fig, ax = plt.subplots()
	ax.scatter(X['AT'], X['V'],c='g',label='example',marker=r'$\clubsuit$')
	ax.set_xlabel('feature:AT')
	ax.set_ylabel('feature:V')
	plt.legend(loc='upper left')
	plt.show()
	theta=lineareg(X,y)
	print("theta are  listed as follow:\n",theta)
