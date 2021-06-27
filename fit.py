# -*- coding: utf-8 -*-
import ycc
import matplotlib.pyplot as plt  
import numpy as np  
import scipy as sp  
from scipy.stats import norm  
from sklearn.pipeline import Pipeline  
from sklearn.linear_model import LinearRegression  
from sklearn.preprocessing import PolynomialFeatures  
from sklearn import linear_model  
 
''''' 数据生成 ''' 
x = np.array(bt1)
y = np.array(bc2)
 
''''' 均方误差根 ''' 
def rmse(y_test, y):  
    return sp.sqrt(sp.mean((y_test - y) ** 2))  
 
''''' 与均值相比的优秀程度，介于[0~1]。0表示不如均值。1表示完美预测.这个版本的实现是参考scikit-learn官网文档  ''' 
def R2(y_test, y_true):  
    return 1 - ((y_test - y_true)**2).sum() / ((y_true - y_true.mean())**2).sum()  
 
 
''''' 这是Conway&White《机器学习使用案例解析》里的版本 ''' 
def R22(y_test, y_true):  
    y_mean = np.array(y_true)  
    y_mean[:] = y_mean.mean()  
    return 1 - rmse(y_test, y_true) / rmse(y_mean, y_true)  
     
fig,ax= plt.subplots(figsize=(6,4),facecolor='w',dpi=400) 
ax.set_ylim(0,1.2)
ax.tick_params(axis='both',color='black',length=3,direction='out')#控制坐标轴刻度显示属性
plt.setp(ax.get_xticklabels(),weight='bold',size=7,style='italic',color='black') #改变ax1实例x轴刻度的大小
plt.setp(ax.get_yticklabels(),weight='bold',size=7,style='italic',color='black') #改变ax1实例x轴刻度的大小
plt.rcParams['font.sans-serif']=['SimHei'] 
ax.set_ylabel('电池健康状态',fontdict=font) #或者plt.ylabel('Scores')
ax.set_xlabel('时间/天',fontdict=font) #或者plt.ylabel('Scores')

plt.scatter(x, y, s=3,c="r")  

degree = [1,2,3,4,5,6]  
y_test = []  
y_test = np.array(y_test)  
 
 
for d in degree:  
    clf = Pipeline([('poly', PolynomialFeatures(degree=d)),  
                    ('linear', LinearRegression(fit_intercept=False))])  
    clf.fit(x[:, np.newaxis], y)  
    y_test = clf.predict(x[:, np.newaxis])  
 
    print(clf.named_steps['linear'].coef_)  
    print('rmse=%.4f, R2=%.4f, R22=%.4f, clf.score=%.4f' %  
      (rmse(y_test, y),  
       R2(y_test, y),  
       R22(y_test, y),  
       clf.score(x[:, np.newaxis], y)))      
 
    plt.plot(x, y_test, linewidth=1)  
 
plt.grid()  
plt.legend(['1','2','3','4','5','6'], loc='best')  
plt.show()  
