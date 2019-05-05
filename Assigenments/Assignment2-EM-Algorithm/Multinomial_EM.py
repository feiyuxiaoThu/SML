# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 20:30:22 2019

@author: feiyuxiao
"""

import numpy as np
from scipy.special import logsumexp
import time

def preprocessor(datasetname,vocabname):
    '''
    Load and process the data
    voc: Vocabulary data
    T : the training data 
    '''
    try:
        f = open(vocabname, "r") 
    except:
        print("File cannot be opened: check your file path or file name",vocabname)
        exit()
    voc = {}
    for line in f.readlines():
        words = line.split('\t')
        voc[int(words[0])] = words[1]
    f.close()
    W = len(voc)
    try:
        F = open(datasetname, "r")  
    except:
        print("File cannot be opened: check your file path or file name",datasetname)
        exit() 
    Lines = F.readlines()      
    Doc_label=[]
    Doc = []
    for line in Lines: 
        Doc_label.append(int(line.split()[0]))
        d = []
        for word in line.split()[1:]:
            d.append(word.split(":"))  
        Doc.append(d)
    F.close()
    
    D = len(Doc_label)
    
    T = np.zeros((D,W))
    for i in range(D):
        for j in range(len(Doc[i])):
            T[i][int(Doc[i][j][0])] = int(Doc[i][j][1])
    
    return voc,T

def fileWrite(arr,k,i):
    '''
    Writing the results to the txt files
    '''
    filePath = "EMresult"+"_"+str(k)+".txt"
    with open(filePath,'a') as f:
        flag=False
        f.write("\n")
        f.write('Frequent words of topic {}:'.format(i+1))
        for temp in arr:
            if flag==True:
                f.write(" "+str(temp))
            else:
                flag=True
                f.write(str(temp))
    print("Writing Finished")
    f.close()


def WordsMostFrequent(mu_log, wordDict,Num=20):
    '''
    Find the most frequent words in each cluster
    Num: Number of words we want to show and default is 10.
    '''
    U = mu_log.transpose()
    (K,W)=U.shape
    word = [[] for i in range(K)]
    print("The K is: %d" %(K))
    for i in range(K):
        list = []
        for j in range(W):
            list.append((U[i,j],j))
        list.sort(key=lambda x:x[0], reverse=True)
        print("-"*50)
        flag = 'Frequent words of topic {}:'.format(i)
        print(flag)
        for j in range(Num):
            word[i].append(wordDict[list[j][1]])
        print(word[i])
        
        fileWrite(word[i],K,i)
        
        print("")
      
'''
Two Normalization functions for EM
'''
def Normalize_array(array):
    sum_array = logsumexp(array)
    return array - np.array([sum_array])

def Normalize_matrix(matrix,axis):
    sum_matrix = logsumexp(matrix,axis=axis)
    if sum_matrix.shape[0] == matrix.shape[0]:
        return (matrix.transpose() - sum_matrix).transpose()
    else:
        return matrix - sum_matrix

'''
Below is the main part: EM 
'''
def initialize(topics_number):
    '''
    Random initialization is better
    '''
    pi = np.random.randint(1, 9, size=topics_number)
    pi_log = np.log(pi) - np.log(pi.sum())
    mu = np.random.randint(1, 9, size=(W, topics_number))
    mu_log = np.log(mu) - np.log(mu.sum(axis=0))
    return pi_log,mu_log
    

def cal_likehood(pi_log,mu_log):
    word_matrix = T
    gamma_un_log = np.dot(word_matrix,mu_log) + pi_log 
    likehood = logsumexp(gamma_un_log,axis=1).sum()
    return likehood

def Expectation(pi_log,mu_log):
    word_matrix = T
    gamma_un_log = np.dot(word_matrix,mu_log) + pi_log 
    gamma_uniform_log = Normalize_matrix(gamma_un_log,axis=1)
    return gamma_uniform_log # gamma_log

def Maximization(gamma_log,topics_number):
    word_matrix = T
    mu_un_log = logsumexp(gamma_log.reshape(D,topics_number,1),
                              b=word_matrix.reshape(D,1,W),axis = 0).transpose() 
    mu_uniform_log = Normalize_matrix(mu_un_log,axis=0)
        
    pi_un_log = logsumexp(gamma_log,axis=0)
    pi_uniform_log = Normalize_array(pi_un_log)
    return pi_uniform_log,mu_uniform_log
       
def EMmain(T,error,Maxiteration,K):
    topics_number = K
    pi_log,mu_log=initialize(topics_number)
    likehood_old = cal_likehood(pi_log,mu_log)
    likehood_change = np.infty
    iteration = 0
    print("main")
    print('Likehood Error Tolerance',error)
    
    while likehood_change >= error or iteration >Maxiteration:
        gamma_log=Expectation(pi_log,mu_log)
        pi_log,mu_log = Maximization(gamma_log,topics_number)
        likehood_new = cal_likehood(pi_log,mu_log)
        likehood_change = likehood_new - likehood_old
        likehood_old = likehood_new           
        iteration +=1
        print("training",iteration,likehood_change)       
    
    WordsMostFrequent(mu_log, voc)
    print("Finished")
        
filename1 = 'nips/nips.libsvm'
filename2 = "nips/nips.vocab"
voc,T = preprocessor(filename1,filename2)
D,W = T.shape
K_list = [5,10,20,30]
error = 1e-10
Maxiteration = 100
for K in K_list:   
    time_start=time.time()
    EMmain(T,error,Maxiteration,K)
    time_end=time.time()
    print('Totally Time Cost',time_end-time_start)

