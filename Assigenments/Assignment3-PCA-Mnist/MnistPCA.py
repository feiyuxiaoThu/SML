import numpy as np
from scipy import io
import matplotlib.pyplot as plt
import scipy.linalg as linalg

'''
X: m x n matrix, for minist, m is the number of images, n is the number of dimensions per image(784)
k: number of principal vectors to compute
'''
def pca(X, k):
	n, p = np.shape(X)
	datamean = np.mean(X, axis = 0)
	eigs = 1.0/n*np.dot(np.transpose(X - datamean),X-datamean)
	vals, vecs = linalg.eigh(eigs, eigvals=(p-k, p-1))
	vals = vals[::-1]
	return datamean, np.transpose(vecs),vals
'''
def pca_uncentered(X, k):
	n, p = np.shape(X)
	sigma = 1.0/n*np.dot(np.transpose(X),X)
	# vecs is kxp matrix
	vals, vecs = linalg.eigh(sigma, eigvals=(p-k, p-1))
	vals = vals[::-1]
	# vecs = np.fliplr(vecs)
	return  np.transpose(vecs),vals
'''

def dimension(percent,Values):
    '''
    values=[]
    for item in Values:
        if abs(item)<0.0000001:
            break
        else:
            values.append(item)
    sumr = np.sum(values)
    '''
    sumr = np.sum(Values)
    
    ratios = np.array([value/sumr for value in Values])
    sumr = ratios
    print(ratios.shape)
    d = 0
    
    ratiosum = 0
    
    for i in range(ratios.size):
        ratiosum = ratiosum + ratios[i]
        sumr[i] = ratiosum
        if ratiosum > percent:
            d = i
            break
    return ratios,d,sumr

def reconstruct(data,V,datamean,indexs):
    for index in indexs:
        origin_pic = data[index,:]
        img = origin_pic.reshape((28,28),order='F')
        m = np.dot(V,origin_pic-datamean)
        n = np.dot(V.T,m) + datamean
        img_re = n.reshape((28,28),order='F')
        plt.figure()
        plt.imshow(img,cmap='gray')
        plt.show()
        plt.figure()
        plt.imshow(img_re,cmap='gray')
        plt.show()

'''       
def reconstruct_un(data,V,indexs):
    for index in indexs:
        origin_pic = data[index,:]
        img = origin_pic.reshape((28,28),order='F')
        m = np.dot(V,origin_pic)
        n = np.dot(V.T,m) 
        img_re = n.reshape((28,28),order='F')
        plt.figure()
        plt.imshow(img,cmap='gray')
        plt.show()
        plt.figure()
        plt.imshow(img_re,cmap='gray')
        plt.show()
''' 

minist = io.loadmat("D:\\Courses\\SML\\Assignments\\Work3\\Code\\mnist-original.mat")
data = minist['data'].T
n, p = np.shape(data)
k = p
Mu, VV ,Values= pca(data, k)
#VV ,Values= pca_uncentered(data, k)
percents = [0.3,0.6,0.9]
for percent in percents:

    ratios,d ,sumr= dimension(percent, Values)
    
    '''
    Show the accuracy and number of principal dimensions relationship
    '''
    plt.figure()
    plt.plot(sumr[:d])
    plt.xlabel('Dimension')
    plt.ylabel('Information')
    plt.title('The accuracy')
    plt.show()
       
    datamean,V ,values=pca(data,d+1)
    #V ,values=pca_uncentered(data,d+1)
        
    indexs = [0,8000]
    
    reconstruct(data,V,datamean,indexs)
    #reconstruct_un(data,V,indexs)


	

