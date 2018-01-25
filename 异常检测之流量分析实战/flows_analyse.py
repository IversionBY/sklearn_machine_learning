import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from collections import Counter
import pprint
#from sklearn.covariance import EllipticEnvelope#调用离群点检测算法
#from sklearn.ensemble import IsolationForest#孤立森林算法异常检测
#from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cross_validation import train_test_split #异常检测算法中对数据集进行拆分
from sklearn.externals import joblib
# from scipy.stats import poisson#导入现成的泊松分布函数


def gathering(picklename):
    '''
    将抓包数据进行汇总为一个矩阵，然后将这个对象持久化
    '''
    dataset = np.zeros((1, 8))
    files = ["下载.csv", "游戏.csv", "游戏2.csv", "码代码.csv", "码代码2.csv"]
    for filename in files:

        data = pd.read_csv(filename, skiprows=1, engine='python')
        data.fillna(value=0, inplace=True)  # 对空值在原来数据集用0来填充
        data = np.array(data)
        dataset = np.vstack((dataset, data))
    datas = dataset[1:, :]
    pickle.dump(datas, open(picklename, "wb"))
    return picklename


def length_analyse():
    '''
    对包的长度的相关参数进行统计和图形化展示
    '''
    try:
        length=pickle.load(open("lengthdata","rb"))
    except Exception:
        datas = pickle.load(open("datas", "rb"))
        print(datas.shape)
        length =list(datas[:, 5])
        pickle.dump(length,open("lengthdata","wb"))
    # 对包长度进行分析
    print("最大包长：", max(length))
    print("最小包长：", min(length))
    print("平均包长：", np.mean(length))
    print("包长方差：", np.var(length))
    
    #画图
    draw_picture(length,"length analyse")


def srcport_analyse():
    '''
    对源端口的分析
    '''
    try:
        srcport=pickle.load(open("srcportdata","rb"))
    except Exception:
        datas = pickle.load(open("datas", "rb"))
        srcport = list(datas[:, 6])
    
    print("最大端口：", max(srcport))
    print("最小端口：", min(srcport))
    #持久化
    pickle.dump(srcport,open("srcportdata","wb"))
    #画图
    draw_picture(srcport,"srcport analyse")


def dstport_analyse():
    '''
    对目的端口的分析
    '''
    try:
        dstport=pickle.load(open("dstportdata","rb"))
    except Exception:
        datas = pickle.load(open("datas", "rb"))
        dstport = list(datas[:, 7])
    
    print("最大端口：", max(dstport))
    print("最小端口：", min(dstport))
    #持久化
    pickle.dump(dstport,open("dstportdata","wb"))
    #画图
    draw_picture(dstport,"dstport analyse")

  


def protocol_analyse():
    '''
    对数据包的协议进行分析，通过一个协议转换字典
    '''
    try:
        protocol=pickle.load(open("protodata","rb"))
    except Exception:
        # 协议映射
        datas = pickle.load(open("datas", "rb"))
        protocols = datas[:, 4]
        Proto_set = set(protocols)

        #为了避免每次生成的映射字典不一样我们这里进行持久化数据载入错误分析
        pro_dic = {}
        for index, pro_name in enumerate(Proto_set, 1):
            # 建立联表,利用enumerate
            pro_dic[pro_name] = index
        pickle.dump(pro_dic,open("protocol_dict","wb"))
        protocol = []
        for i in protocols:
            protocol.append(pro_dic[i])
        #持久化
        pickle.dump(protocol,open("protodata","wb"))
    
    pro_dic=pickle.load(open("protocol_dict","rb"))
    #映射字典展示  
    print("转换字典\n", pro_dic, "\n\n\n")
    search_dic = dict(zip(pro_dic.values(), pro_dic.keys())) 
    #反向建立查询表
    print("查询字典\n", search_dic, "\n\n\n")
    
    # 统计协议数量
    '''
    change_dcount = {}
    dcount = Counter(protocol)
    for key in dcount.keys():
        change_dcount[search_dic[key]] = dcount[key]
        '''
    
    #画图
    draw_picture(protocol,"protocol analyse")
    
 

def srcip():
    '''
    对源IP的分析
    '''
    try:
        srcips=pickle.load(open("srcipdata","rb"))
    except Exception:
        datas = pickle.load(open("datas", "rb"))
        srcipdatas = datas[:, 2]
        srcip_set = set(srcipdatas)
        srcip_dic={}
        for index, srcip  in enumerate(srcip_set, 1):
            # 建立联表,利用enumerate
            srcip_dic[srcip] = index
        pickle.dump(srcip_dic,open("srcip_dict","wb"))
        srcip_changed_data = []
        for i in srcipdatas:
            srcip_changed_data.append(srcip_dic[i])
        #持久化
        pickle.dump(srcip_changed_data,open("srcipdata","wb"))
        
    #映射字典展示 
    srcip_dic=pickle.load(open("srcip_dict","rb")) 
    print("转换字典\n",srcip_dic ,"\n\n\n")
    search_dic = dict(zip(srcip_dic.values(), srcip_dic.keys())) 
    #反向建立查询表
    print("查询字典\n", search_dic, "\n\n\n")
    
    draw_picture(srcips,"srcip analyse")


  
def dstip():
    '''
    对目的IP的分析
    '''


def draw_picture(dataset,item):
    '''
    画出直方图和箱线图
    '''
    #直方图
    plt.title(item)
    plt.hist(dataset,100,alpha=0.5)
    #plt.axis(range(min(dataset),max(dataset),1))
    plt.grid(True)
    plt.show()

    #箱线图
    plt.title(item)
    plt.boxplot(x = dataset, # 指定绘图数据
            patch_artist=True, # 要求用自定义颜色填充盒形图，默认白色填充
            showmeans=True, # 以点的形式显示均值
            boxprops = {'color':'blue','facecolor':'#9999ff'}, # 设置箱体属性，填充色和边框色
            flierprops = {'marker':'o','markerfacecolor':'red','color':'red'}, # 设置异常值属性，点的形状、填充色和边框色
            meanprops = {'marker':'D','markerfacecolor':'indianred'}, # 设置均值点的属性，点的形状、填充色
            medianprops = {'linestyle':'--','color':'orange'}) # 设置中位数线的属性，线的类型和颜色
    plt.show()
    
    '''old code

    x = p['fliers'][0].get_xdata() # 'flies'即为异常值的标签.
    y = p['fliers'][0].get_ydata()
    y.sort() #从小到大排序

    for i in range(len(x)): 
        if i>0:
            plt.annotate(y[i], xy = (x[i],y[i]), xytext=(x[i]+0.05 -0.8/(y[i]-y[i-1]),y[i]))
        else:
            plt.annotate(y[i], xy = (x[i],y[i]), xytext=(x[i]+0.08,y[i]))
    
    plt.tick_params(top='off', right='off') 
    '''

def Anomaly_Detection():
    '''
    调用机器学习sklearn库来对数据进行异常诊断
    '''
    #数据整合
    pickle_datalist=["srcipdata","protodata","srcportdata","lengthdata","dstportdata"]
    X=np.zeros((657254,5))
    i=0
    for name in pickle_datalist:
        X[:,i]=pickle.load(open(name,"rb"))
        i+=1
    #拟合
    X_train, X_test = train_test_split(X, random_state=10)#拆分成训练集和测试集
    clf = LocalOutlierFactor()
    clf.fit(X_train)
    y_pred_test = clf.fit_predict(X)#fit_predict是针对给出的数据来拟合然后判断误差值，所以不同的预测集返回也会不一样
    #对预测异常的数据进行处理
    predicdata=np.zeros((1,5))
    num=0
    for i in y_pred_test:
        if i==-1:
            predicdata = np.vstack((predicdata, X[num,:]))        
        num=num+1
    pickle.dump(predicdata,open("predicdata","wb"))
    datas=pickle.load(open("predicdata","rb"))
    print(len(datas))
    print(datas[0:5,:])








if __name__ == "__main__":
    #gathering("datas")
    #length_analyse()
    srcport_analyse()
    #dstport_analyse()
    #protocol_analyse()
    #srcip()
    #Anomaly_Detection()
    