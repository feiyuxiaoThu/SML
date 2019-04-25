# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 14:29:26 2019
@author: feiyuxiao
"""

from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from scipy.sparse import csr_matrix

clf = LogisticRegression()
SGD_clf = SGDClassifier(max_iter = 1000)


X_train, y_train = load_svmlight_file("a9a/a9a")
X_test , y_test = load_svmlight_file("a9a/a9a.t")

X_m = X_test.todense()

row_xtest = X_m.shape[0]

line = np.zeros(row_xtest)

XX_test = np.column_stack((X_m,line))

print(XX_test.shape)
X_test = csr_matrix(XX_test)

clf.fit(X_train,y_train)
SGD_clf.fit(X_train,y_train)

y_pred = clf.predict(XX_test)
SGD_y_pred = SGD_clf.predict(XX_test)

y_pred_training = clf.predict(X_train)
SGD_y_pred_training = SGD_clf.predict(X_train)

print('For logisticRegeression')

print('Error in training set',accuracy_score(y_train,y_pred_training))
print('Error in test set ',accuracy_score(y_test,y_pred))

print("For SGD LR")
print('Error in training set',accuracy_score(y_train,SGD_y_pred_training))
print("Error in test set",accuracy_score(y_test,SGD_y_pred))