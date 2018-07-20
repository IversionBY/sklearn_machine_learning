import re
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split#进行测试集和训练集的拆分
from sklearn import metrics#导入metrics模块，对模型结果进行评估
from keras.preprocessing.sequence import pad_sequences#导入将列表转换为numpy的pod_sequences
from keras.layers import Input, LSTM, Dense,Embedding,Activation
from keras.models import Sequential
from keras.models import Model#引入model来保存模型
#from keras.utils import plot_model#绘制模型图

max_sequences_len=300#设置数据的字符串列表的默认长度
max_sys_call=0

def load_one_flle(filename):
    global max_sys_call
    x=[]
    with open(filename) as f:
        line=f.readline()
        line=line.strip('\n')
        line=line.split(' ')
        for v in line:
            if len(v) > 0:
                x.append(int(v))
                if int(v) > max_sys_call:
                    max_sys_call=int(v)
    return x

def load_adfa_training_files(rootdir):
    x=[]
    y=[]
    list = os.listdir(rootdir)
    for i in range(0, len(list)):
        path = os.path.join(rootdir, list[i])
        if os.path.isfile(path):
            x.append(load_one_flle(path))
            y.append(0.)
    return x,y

def dirlist(path, allfile):
    filelist = os.listdir(path)
    for filename in filelist:
        filepath = os.path.join(path, filename)
        if os.path.isdir(filepath):
            dirlist(filepath, allfile)
        else:
            allfile.append(filepath)
    return allfile

def load_adfa_webshell_files(rootdir):
    x=[]
    y=[]
    allfile=dirlist(rootdir,[])
    for file in allfile:
        if re.match(r"./data/ADFA-LD/Attack_Data_Master/Web_Shell*",file):
            x.append(load_one_flle(file))
            y.append(1.)
    return x,y

def do_rnn(trainX, testX, trainY, testY):
    global max_sequences_len
    global max_sys_call
    #将x,y分别转换为numpy型
    trainX = pad_sequences(trainX, maxlen=max_sequences_len, value=0.)
    testX = pad_sequences(testX, maxlen=max_sequences_len, value=0.)
    trainY = to_categorical(trainY)
    testY_old=testY#将一维的数据保存，方便后面用于评估
    testY = to_categorical(testY)
    
    #构建神经网络模型
    model = Sequential()
    model.add(Embedding(input_dim=max_sys_call+1,output_dim=128, input_length=max_sequences_len))#input_dim标识的是词量大小，在这里表示的是数据里面的ascii值，由于存在一些干扰项，所以把词量大小设置成max_sys_call
    model.add(LSTM(100,dropout=0.04,activation='tanh'))
    model.add(Dense(2,activation="softmax"))
    model.summary() # 打印模型
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    model.fit(trainX,trainY)
    #plot_model(model, to_file='model.png')
    model.save(r"./model")
    y_predict_list=model.predict(testX)
    y_predict = []
    for i in y_predict_list:
        #print  i[0]
        if i[0] > 0.5:
            y_predict.append(0)
        else:
            y_predict.append(1)

    #模型评估
    print(metrics.recall_score(testY_old, y_predict))
    print(metrics.accuracy_score(testY_old, y_predict))
    print(metrics.f1_score(testY_old, y_predict))

    


if __name__ == '__main__':
    x1,y1=load_adfa_training_files(r"./data/ADFA-LD/Training_Data_Master/")#载入正确的文本数据
    x2,y2=load_adfa_webshell_files(r"./data/ADFA-LD/Attack_Data_Master/")#载入webshell数据
    x=x1+x2
    y=y1+y2
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    do_rnn(x_train, x_test, y_train, y_test)