"""
==========================================
IsolationForest example
==========================================

An example using :class:`sklearn.ensemble.IsolationForest` for anomaly
detection.

The IsolationForest 'isolates' observations by randomly selecting a feature
and then randomly selecting a split value between the maximum and minimum
values of the selected feature.

Since recursive partitioning can be represented by a tree structure, the
number of splittings required to isolate a sample is equivalent to the path
length from the root node to the terminating node.

This path length, averaged over a forest of such random trees, is a measure
of normality and our decision function.

Random partitioning produces noticeable shorter paths for anomalies.
Hence, when a forest of random trees collectively produce shorter path lengths
for particular samples, they are highly likely to be anomalies.

"""

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import dill

rng = np.random.RandomState(42)#due to the results are random each time, we set the seeds for pseudorandom number generator

##########load normal data
data = scipy.io.loadmat('datanormal.mat')  # 读取mat文件
fuel=data['fuel']
t1=data['t1']
p2=data['p2']
t4=data['t4']
power=data['p']
data1 = scipy.io.loadmat('rh_normal.mat')  # 读取mat文件
rh=data1['rhd']
#a=t1[0:100,:]
'''x=[]
x.append(t1)
print(x)'''
#https://blog.csdn.net/qq_39516859/article/details/80666070
x=np.hstack((t1,fuel,p2,t4,power,rh))#将多个行向量合并成矩阵,normal data
a,b=x.shape#a row b column



############load abnormal data
dataab = scipy.io.loadmat('dataab1.mat')  # 读取mat文件
fuelab=dataab['fuelab1']
t1ab=dataab['t1ab1']
p2ab=dataab['p2ab1']
#二维数组shape 9128,1
t4ab=dataab['t4ab1']
powerab=dataab['pab1']
data2= scipy.io.loadmat('rh_abnormal.mat')  # 读取mat文件
rhab=data2['rhab1d']
#xab=np.hstack((t1ab[:,0],fuelab[:,0],p2ab,t4ab[:,0],powerab[:,0]))
#该方法直接将各个Row vector连接为一个大的行向量，不知道为什么
x_outlier=np.hstack((t1ab,fuelab,p2ab,t4ab,powerab,rhab))
'''t1ab[:,0].shape
fuelab[:,0].shape'''


#####generate training data and test data
x_train=x[0:int(a*0.7),:]
#https://blog.csdn.net/songyunli1111/article/details/79322103
#min-max standardization for training data
xmin=np.min(x_train,axis=0)
xmax=np.max(x_train,axis=0)
x_trains=(x_train-xmin)/(xmax-xmin)
'''x_train[-1,:]
x[int(a/2)-1,:]'''

x_val=x[int(a*0.7):int(a*0.85),:]#数组索引时，后一个元素不包含，int(a/2):a,
#即不包含第a个元素;a为行数，python从0开始，故数组中最后一个元素为第a-1个元素
x_vals=(x_val-xmin)/(xmax-xmin)#standalization of test data

x_test=x[int(a*0.85):a,:]#数组索引时，后一个元素不包含，int(a/2):a,
#即不包含第a个元素;a为行数，python从0开始，故数组中最后一个元素为第a-1个元素
x_tests=(x_test-xmin)/(xmax-xmin)#standalization of test data

x_outliers=(x_outlier-xmin)/(xmax-xmin)


# fit the model
clf = IsolationForest(behaviour='new', max_samples=8,
                      random_state=rng,contamination='auto')
'''clf = IsolationForest(behaviour='new', max_samples=100,
                      random_state=rng, contamination='auto')'''
clf.fit(x_trains)
y_pred_train = clf.predict(x_trains)
y_pred_val = clf.predict(x_vals)
y_pred_test = clf.predict(x_tests)
y_pred_outliers = clf.predict(x_outliers)

########computing classification accuracy
n_error_train = y_pred_train[y_pred_train == -1].size
acc_train=1-n_error_train/len(x_train)
n_error_val = y_pred_val[y_pred_val == -1].size
acc_val=1-n_error_val/len(x_val)
n_error_test = y_pred_test[y_pred_test == -1].size
acc_test=1-n_error_test/len(x_test)
n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size
acc_outliers=1-n_error_outliers/len(x_outliers)

#save data_all.mat in python  https://www.jb51.net/article/135384.htm
scipy.io.savemat('data_all.mat',{'x_train':x_train,'x_val':x_val,'x_test':x_test,'x_outlier':x_outlier}) 

###########save results
dill.dump_session('IsolationForest_egt.pkl')#save all results 

####load results
#dill.load_session('IsolationForest_egt.pkl')