# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 18:43:06 2019
@author: feiyuxiao
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split

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


def IRLS (dataX,datalabel,Lambda,epsilon = 0.00001,max_iter = 20):
    m,n = np.shape(dataX)
       
    ratios = np.zeros(n)
    
    times_iter = 0
    
    w_norm = [0] # 
    
    w_error_norm = []
    
    for x in range(max_iter):
        temp0 =dataX.dot(ratios)
        Y = sigmoid(temp0)
        
        R = Y*(1.0-Y)
       
        Rnn = np.diag(R)
        
        matrix0 = dataX.transpose().dot(Rnn)
     
        matrix1 = matrix0.dot(dataX)+Lambda*np.eye(n,n)
         
        matrix2 = matrix0.dot(temp0+ (datalabel-Y)/R)
        ratios_update = np.linalg.solve(matrix1,matrix2)
        
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
    return ratios,times_iter,w_norm,w_error_norm #,results
   
def accuracy(dataX,datalabel,weights_pre):
    y_pre = np.zeros(datalabel.shape[0])
    y_pre = sigmoid(dataX.dot(weights_pre))
    flag = 0
    for i in range(datalabel.shape[0]):
        if (abs(y_pre[i]-datalabel[i]) < 0.5):
            flag+=1

    accuracy_score = flag/datalabel.shape[0]
   
    return accuracy_score    

# main
file_train = "a9a/a9a"
file_test = "a9a/a9a.t"


train , target = loadData(file_train)
X_test , y_test = loadData(file_test)

#train = train[:5000,:]
#target = target[:5000]


X_train,test_x, train_y, test_y = train_test_split(train,
                                                   target,
                                                   test_size = 0.2,
                                                   random_state = 0)

row_xtest = X_test.shape[0]

line = np.zeros(row_xtest)

X_test = np.column_stack((X_test,line))

lambda_list = [0.01,0.1,0.5,1]

ratios = []
accuracy_train = []
accuracy_test = []
time = []
w_norm=[]
w_error_norm=[]
epsilon = 0.00001
max_iter = 20

print("Training for weights")  
for Lambda in lambda_list:    
    ratio,_time,wnorm,werrornorm = IRLS(X_train,train_y,Lambda)
    w_norm.append(wnorm)
    w_error_norm.append(werrornorm)
    ratios.append(ratio)
    time.append(_time)
    accuracy_score_test = accuracy(test_x,test_y,ratio)       
    accuracy_test.append(accuracy_score_test)


chioce_Lambda = accuracy_test.index(max(accuracy_test)) 
accuracy_final = accuracy(X_test , y_test,ratios[chioce_Lambda]) 
 
print("finished")
print("Best lambda",lambda_list[chioce_Lambda],"accuracy",accuracy_test[chioce_Lambda])
print("Accuracy in test set",accuracy_final)

plt.figure()
plt.plot(lambda_list,time,"o-")
plt.xlabel("Lambda")
plt.ylabel("times")
plt.title("Literation times")
plt.savefig("1.png",dpi=200)

plt.figure()
plt.plot(w_norm[chioce_Lambda],"o-")
plt.xlabel("Iteration")
plt.ylabel("w_norm")
plt.title("weights norm")
plt.savefig("2.png",dpi=200)

plt.figure()
plt.plot(w_error_norm[chioce_Lambda],"o-")
plt.xlabel("Iteration")
plt.ylabel("w_error_norm")
plt.title("weights error norm")
plt.savefig("3.png",dpi=200)

plt.figure()
plt.plot(lambda_list,accuracy_test,"o-")
plt.xlabel("Lambda")
plt.ylabel("accuracy")
plt.title("Prediction Accuracy")
plt.savefig("4.png",dpi=200)