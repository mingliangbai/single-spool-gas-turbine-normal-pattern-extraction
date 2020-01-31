# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 20:23:16 2019

@author: baimingliang
"""
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from sklearn import svm
import dill

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

acc_train=[]
acc_val=[]
acc_test=[]
acc_outliers=[]
#########fit the model
for u in np.arange(0.1,2.1,0.1):
    clf = svm.OneClassSVM(nu=0.01, kernel="rbf", gamma=u)#nu为训练集上的正常样本在分类后被划分为异常样本的比例
    clf.fit(x_trains)
    y_pred_train = clf.predict(x_trains)
    y_pred_val = clf.predict(x_vals)
    y_pred_test = clf.predict(x_tests)
    y_pred_outliers = clf.predict(x_outliers)
    ########computing classification accuracy
    n_error_train = y_pred_train[y_pred_train == -1].size
    acc_train.append(1-n_error_train/len(x_train))
    n_error_val = y_pred_val[y_pred_val == -1].size
    acc_val.append(1-n_error_val/len(x_val))
    n_error_test = y_pred_test[y_pred_test == -1].size
    acc_test.append(1-n_error_test/len(x_test))
    n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size
    acc_outliers.append(1-n_error_outliers/len(x_outliers))

#%%
clf = svm.OneClassSVM(nu=0.01, kernel="linear")#nu为训练集上的正常样本在分类后被划分为异常样本的比例
clf.fit(x_trains)
y_pred_train = clf.predict(x_trains)
y_pred_val = clf.predict(x_vals)
y_pred_test = clf.predict(x_tests)
y_pred_outliers = clf.predict(x_outliers)
########computing classification accuracy
n_error_train = y_pred_train[y_pred_train == -1].size
acc_trainlin=(1-n_error_train/len(x_train))
n_error_val = y_pred_val[y_pred_val == -1].size
acc_vallin=(1-n_error_val/len(x_val))
n_error_test = y_pred_test[y_pred_test == -1].size
acc_testlin=(1-n_error_test/len(x_test))
n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size
acc_outlierslin=(1-n_error_outliers/len(x_outliers))

#%%
clf = svm.OneClassSVM(nu=0.01, kernel="sigmoid")#nu为训练集上的正常样本在分类后被划分为异常样本的比例
clf.fit(x_trains)
y_pred_train = clf.predict(x_trains)
y_pred_val = clf.predict(x_vals)
y_pred_test = clf.predict(x_tests)
y_pred_outliers = clf.predict(x_outliers)
########computing classification accuracy
n_error_train = y_pred_train[y_pred_train == -1].size
acc_trainsig=(1-n_error_train/len(x_train))
n_error_val = y_pred_val[y_pred_val == -1].size
acc_valsig=(1-n_error_val/len(x_val))
n_error_test = y_pred_test[y_pred_test == -1].size
acc_testsig=(1-n_error_test/len(x_test))
n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size
acc_outlierssig=(1-n_error_outliers/len(x_outliers))

#%%
acc_trainp=[]
acc_valp=[]
acc_testp=[]
acc_outliersp=[]
for d in np.arange(2,11,1):
    clf = svm.OneClassSVM(nu=0.05, kernel="poly", degree=d)#nu为训练集上的正常样本在分类后被划分为异常样本的比例
    #when nu=0.01, the performance is very poor
    clf.fit(x_trains)
    y_pred_train = clf.predict(x_trains)
    y_pred_val = clf.predict(x_vals)
    y_pred_test = clf.predict(x_tests)
    y_pred_outliers = clf.predict(x_outliers)
    ########computing classification accuracy
    n_error_train = y_pred_train[y_pred_train == -1].size
    acc_trainp.append(1-n_error_train/len(x_train))
    n_error_val = y_pred_val[y_pred_val == -1].size
    acc_valp.append(1-n_error_val/len(x_val))
    n_error_test = y_pred_test[y_pred_test == -1].size
    acc_testp.append(1-n_error_test/len(x_test))
    n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size
    acc_outliersp.append(1-n_error_outliers/len(x_outliers))

#%%
########save results
dill.dump_session('one_svm_egt.pkl')#save all results 

####load results
#dill.load_session('one_svm_egt.pkl')

#for i in range(0,3,1):
#    print(i)