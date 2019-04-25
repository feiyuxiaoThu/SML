# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 18:43:06 2019

@author: feiyuxiao
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_svmlight_file
import scipy.io as sio


import matlab.engine
eng = matlab.engine.start_matlab()


def loadData(filename):
    X,Y = load_svmlight_file(filename)
    X = X.todense()
    X = np.array(X)
    Y = (Y+1)/2
    for y in Y:
        y = round(y)   # to resize to {0,1}  
    return X,Y


def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))
 
def IRLS (dataX,datalabel,epsilon = 0.00001,max_iter = 20):
    m,n = np.shape(dataX)
       
    ratios = np.zeros(n)
    ratios_update = np.zeros(n)
    
    times_iter = 0
    
    w_norm = [0] 
    
    w_error_norm = []
    for x in range(max_iter):
        
        temp0 =dataX.dot(ratios)
        Y = sigmoid(temp0)
        
        R = Y*(1.0-Y)
       
        Rnn = np.diag(R)
        
        matrix0 = dataX.transpose().dot(Rnn)
        temp1 = matrix0.dot(dataX)
           
        # Use matlab to solve the ill contional equation
        temp2 = matrix0.dot(temp0+ (datalabel-Y)/R)
        sio.savemat("equation.mat",{"A":temp1,"b":temp2})
        double_ratios = eng.matsolve()
        double_ratios = np.array(double_ratios)
        
        ratios_update[:] =  double_ratios[:,0]
                
        wnorm = np.linalg.norm(ratios_update)
        w_norm.append(wnorm)
        
        error = ratios_update - ratios
       
        error_norm = np.linalg.norm(error)
        w_error_norm.append(error_norm)
         
        ratios = ratios_update

        if error_norm < epsilon:
            break
        ratios = ratios_update    
        
        times_iter += 1
    return ratios,times_iter,w_norm,w_error_norm 
   
def accuray(dataX,datalabel,weights_pre):
    y_pre = np.zeros(datalabel.shape[0])
    y_pre = sigmoid(dataX.dot(weights_pre))
    flag = 0
    for i in range(datalabel.shape[0]):
        if (abs(y_pre[i]-datalabel[i]) < 0.5):
            flag+=1

    accuracy_score = flag/datalabel.shape[0]
   
    return accuracy_score    

# main
#file_train = "a9a"
file_train = "a9a/a9a"
file_test = "a9a/a9a.t"


X_train , y_train = loadData(file_train)
X_test , y_test = loadData(file_test)

row_xtest = X_test.shape[0]

line = np.zeros(row_xtest)

X_test = np.column_stack((X_test,line))

time = []
w_norm=[]
w_error_norm = []
epsilon = 0.00001
max_iter = 20


print("Training for weights")  
ratio,_time,w_norm,w_error_norm = IRLS(X_train,y_train)

time.append(_time)
accuracy_train = accuray(X_train,y_train,ratio)

accuracy_test = accuray(X_test,y_test,ratio)       


print("finished")

print("times_iter",_time)
print("accuracy in train set",accuracy_train)  
print("accuracy in test set",accuracy_test)


plt.figure()
plt.plot(w_norm,"o-")
plt.xlabel("Iteration")
plt.ylabel("w_norm")
plt.title("weights norm")
plt.savefig("01.png",dpi=200)

plt.figure()
plt.plot(w_error_norm,"o-")
plt.xlabel("Iteration")
plt.ylabel("w_error_norm")
plt.title("weights error norm")
plt.savefig("02.png",dpi=200)
