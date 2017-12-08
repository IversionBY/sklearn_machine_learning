# -*- coding: utf-8 -*-
# @Author: IversionBY
# @Date:   2017-11-29 19:40:24
# @Last Modified by:   IversionBY
# @Last Modified time: 2017-12-08 12:47:40

'''
本节数据是UCI上面的葡萄酒数据集，来实现对三种型号的葡萄酒分类
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split#数据集拆分
from sklearn import metrics
from sklearn.metrics import mean_squared_error,precision_score,recall_score,f1_score#导入计算均方等属性的函数
from sklearn.model_selection import cross_val_predict#导入交叉验证函数
from sklearn.preprocessing import StandardScaler#导入标准化函数
from sklearn.linear_model import RidgeCV,LogisticRegression#导入岭回归实现进行正则参数的选择,逻辑回归进行拟合
from sklearn.learning_curve import learning_curve,validation_curve #学习曲线,验证曲线
from sklearn.pipeline import Pipeline#导入流水线作业
from sklearn.externals import joblib#模型持久化

def logisticReg(X,y,alpha_set):

	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10)#拆分成训练集和测试集
	#规范化参数
	stdsc=StandardScaler()
	X_train_std=stdsc.fit_transform(X_train)
	X_test_std=stdsc.transform(X_test)
	#岭回归计算最优alpha
	LR2=RidgeCV(alphas=alpha_set)#选择惩罚参数
	LR2.fit(X_train_std,y_train)

	#对选好的参数进行逻辑回归
	LR1=LogisticRegression(penalty='l1',C=LR2.alpha_)
	joblib.dump(LR1, "train_model_logisticregression.m")#持久化模型
	LR1.fit(X_train_std,y_train)
	y_pred=LR1.predict(X_test_std)
	#------------------------------------------------------------------------图形化评估，学习曲线和验证曲线
	#绘制学习曲线
	learning_cur(X_train_std,y_train)
	#绘制正则参数验证曲线
	cv_curve(X_train_std,y_train,alphas)
	#-------------------------------------------------------------------------拟合效果的评估参数
	#预测的正确率
	print("trainning accuracy:",LR1.score(X_train_std,y_train))
	# 测试集用scikit-learn计算MSE,RMSE
	print("test_MSE:",metrics.mean_squared_error(y_test, y_pred))
	print ("test_RMSE:",np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
	#十折交叉集检验,本编程采用k折法，而NG的课堂告诉我们他采用的是hand out法则
	predicted = cross_val_predict(LR1, X, y, cv=10,n_jobs=1)
	print ("CV_MSE:",metrics.mean_squared_error(y, predicted))
	print ("CV_RMSE:",np.sqrt(metrics.mean_squared_error(y, predicted)))
	#计算PRE,REC,F1
	print("precision:%.4f"%precision_score(y_true=y_test,y_pred=y_pred,average='macro'))
	print("recall:%.4f"%recall_score(y_true=y_test,y_pred=y_pred,average='micro'))
	print("F1:%.4f"% f1_score(y_true=y_test,y_pred=y_pred,average='weighted'))
	#average后面的参数分别是多类别分类中的三种计算指标或者说方式
	return LR1.intercept_,LR1.coef_


def learning_cur (train_x,train_y):
	'''
	学习曲线要求传入两个规范化的训练集的X,y
	'''
	train_sizes,train_scores,test_scores=learning_curve(estimator=LogisticRegression(penalty='l2'),X=train_x,y=train_y,train_sizes=np.linspace(0.1,1,20),cv=5,n_jobs=1)
	train_mean=np.mean(train_scores,axis=1)
	train_std=np.std(train_scores,axis=1)
	test_mean=np.mean(test_scores,axis=1)
	test_std=np.std(test_scores,axis=1)
	plt.plot(train_sizes,train_mean,color='blue',marker='o',markersize=5,label='training accuracy')
	plt.fill_between(train_sizes,train_mean+train_std,train_mean-train_std,alpha=0.2,color='blue')
	plt.plot(train_sizes,test_mean,color='yellow',marker='s',linestyle='--',markersize=5,label='validation accuracy')
	plt.fill_between(train_sizes,test_mean+test_std,test_mean-test_std,alpha=0.2,color='yellow')
	plt.grid()
	plt.xlabel('numble of traininf sample')
	plt.ylabel('Accuracy')
	plt.legend(loc='lower right')
	plt.ylim(0.8,1.0)
	plt.show()

def cv_curve(train_x,train_y,alphas):
	'''
	验证曲线需要传入一个标准化的X,y以及参数列表
	'''
	pip=Pipeline([('clf',LogisticRegression(penalty='l2'))])

	train_scores,test_scores=validation_curve(estimator=pip,X=train_x,y=train_y,param_name='clf__C',param_range=alphas,cv=10)
	train_mean=np.mean(train_scores,axis=1)
	train_std=np.std(train_scores,axis=1)
	test_mean=np.mean(test_scores,axis=1)
	test_std=np.std(test_scores,axis=1)
	plt.plot(alphas,train_mean,color='blue',marker='o',markersize=5,label='training accuracy')
	plt.fill_between(alphas,train_mean+train_std,train_mean-train_std,alpha=0.2,color='blue')
	plt.plot(alphas,test_mean,color='yellow',marker='s',linestyle='--',markersize=5,label='validation accuracy')
	plt.fill_between(alphas,test_mean+test_std,test_mean-test_std,alpha=0.2,color='yellow')
	plt.grid()
	plt.xscale('log')
	plt.xlabel('Parameter C')
	plt.ylabel('Accuracy')
	plt.legend(loc='lower right')
	plt.ylim(0.8,1.0)
	plt.show()



if __name__=="__main__":
	'''
	对数据进行初步处理,也即数据获取和X，y变量的切片
	'''
	URL='https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'
	wine_dataset=pd.read_csv(URL,header=None)
	wine_dataset.columns=['class label','F1','F2','F3','F4','F5','F6','F7','F8','F9','F10','F11','F12','F13']
	X,y=wine_dataset.iloc[:,1:].values,wine_dataset.iloc[:,0].values
	alphas=[0.001, 0.01,0.5, 1,5, 10, 50,100, 1000,10000,100000]#传入的正则化参数，函数里面将使用岭回归来选取最优alpha
	weight0,weights=logisticReg(X,y,alphas)



