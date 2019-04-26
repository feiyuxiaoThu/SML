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
def loadData(libfilepath, vocfilepath):
    print("Loading data......")
    X = []
    #Loading word file
    wordDict = {}
    vocfile = open(vocfilepath,"r")
    for line in vocfile.readlines():
        words = line.split('\t')
        wordDict[int(words[0])] = words[1]
    vocfile.close()
    # Loading lib file
    wordSize = len(wordDict)
    libfile = open(libfilepath,"r")
    for line in libfile.readlines():
        doc = np.zeros(wordSize, dtype='float')
        words = line.strip().split('\t')[1].split(" ")
        for word in words:
            word = word.split(":")
            doc[int(word[0])] = int(word[1])
        X.append(doc)
    libfile.close()
    return np.asarray(X), wordDict

def fileWrite(arr,k,i):
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
class EM:
    '''
        word_matrix: word occurrence matrix for the corpus
    '''
    def __init__(self,topics_number = 5,error = 1e-10,maxiteration=100):
        self.topics_number = topics_number
        self.pi = None
        self.mu = None
        self.er = error
        self.maxiteration = maxiteration
    
    def cal_likehood(self):
        '''
        gamma_un_log: log of gamma (Not uniform)
        And this matrix is D * K
        '''
        word_matrix = self.word_matrix
        D,W = word_matrix.shape
        pi_log = self.pi_log
        mu_log = self.mu_log
        gamma_un_log = np.dot(word_matrix,mu_log) + pi_log 
        likehood = logsumexp(gamma_un_log,axis=1).sum()
        return likehood
    
    
    def initialize(self):
        word_matrix = self.word_matrix
        topics_number = self.topics_number
        W = word_matrix.shape[1]
        '''
        Using random initialization to get better results
        '''
        pi = np.random.randint(1, 9, size=topics_number)
        pi_log = np.log(pi) - np.log(pi.sum())
        mu = np.random.randint(1, 9, size=(W, topics_number))
        mu_log = np.log(mu) - np.log(mu.sum(axis=0))
  
        self.pi_log = pi_log
        self.mu_log = mu_log
        
    def Expectation(self):
        '''
        gamma_uniform_log: gamma_log which is uniform
        '''
        word_matrix = self.word_matrix
        pi_log = self.pi_log
        mu_log = self.mu_log
        gamma_un_log = np.dot(word_matrix,mu_log) + pi_log 
        gamma_uniform_log = Normalize_matrix(gamma_un_log,axis=1)
        
        self.gamma_log = gamma_uniform_log
        
    def Maximization(self):
        word_matrix = self.word_matrix
        gamma_log = self.gamma_log
        topics_number = self.topics_number
        D,W = word_matrix.shape[0],word_matrix.shape[1]
        mu_un_log = logsumexp(gamma_log.reshape(D,topics_number,1),
                              b=word_matrix.reshape(D,1,W),axis = 0).transpose() # !!!!
        mu_uniform_log = Normalize_matrix(mu_un_log,axis=0)
        
        pi_un_log = logsumexp(gamma_log,axis=0)
        pi_uniform_log = Normalize_array(pi_un_log)
        
        self.pi_log = pi_uniform_log
        self.mu_log = mu_uniform_log
    
    def main(self,word_matrix):
        self.word_matrix = word_matrix
        self.initialize()
        likehood_old =  self.cal_likehood()
        likehood_change = np.infty
        iteration = 0
        print("main")
        print('Likehood Error Tolerance',self.er)
        mu_log = None
        while likehood_change >= self.er or iteration >self.maxiteration:
            self.Expectation()
            self.Maximization()
            mu_log = self.mu_log
            likehood_new = self.cal_likehood()
            likehood_change = likehood_new - likehood_old
            likehood_old = likehood_new           
            iteration +=1
            print("training",iteration,likehood_change)       
        
        WordsMostFrequent(mu_log, voc)
        print("Finished")
        
filename1 = 'nips/nips.libsvm'
filename2 = "nips/nips.vocab"
voc,T = preprocessor(filename1,filename2)
K_list = [5,10,20,30]
error = 1e-10
Maxiteration = 100
for K in K_list:   
    time_start=time.time()
    trainingmethod = EM(K,error,Maxiteration)
    trainingmethod.main(T)
    time_end=time.time()
    print('Totally Time Cost',time_end-time_start)

