# -*- coding: utf-8 -*-
# @Author: IversionBY
# @Date:   2017-12-04 15:17:35
# @Last Modified by:   IversionBY
# @Last Modified time: 2017-12-08 16:11:49

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.image as mpimg
from PIL import Image
from pylab import *


def PCA_algorithmic(dataset,K=0.95):

	#dateset: which is expected like a picture data;
	#K:the values you want to decrese into,if not,k may be set automatically with a accuracy rate 95%
	pca=PCA(n_components=K)
	New_data=pca.fit_transform(dataset)
	print()
	return New_data
	

if __name__=="__main__":

	img=np.array(Image.open("06-1m.jpg").convert("L"))#使用PIL打开图像灰化后存入np对象
	
	#waiting for refresh......