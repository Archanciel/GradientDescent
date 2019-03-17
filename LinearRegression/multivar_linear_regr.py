
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

my_data = pd.read_csv('home.txt',names=["size","bedroom","price"])

#we need to normalize the features using mean normalization
my_data = (my_data - my_data.mean())/my_data.std()
print(my_data.head())


#setting the matrixes
X = my_data.iloc[:,0:2]
ones = np.ones([X.shape[0],1])
X = np.concatenate((ones,X),axis=1)
print(X)

y = my_data.iloc[:,2:3].values #.values converts it from pandas.core.frame.DataFrame to numpy.ndarray
print('y: ', y)
theta = np.zeros([1,3])

print('theta: ', theta)

#computecost
def computeCost(X,y,theta):
    tobesummed = np.power(((X @ theta.T)-y),2)
    return np.sum(tobesummed)/(2 * len(X))

def gradientDescent(X,y,theta,iters,alpha):
    cost = np.zeros(iters)
    N = len(X)
    for i in range(iters):
        fPrimeG = np.dot(X, theta.T) - y
        g = np.dot(fPrimeG.flatten(), X) / N

        grad = np.sum(X * (X @ theta.T - y), axis=0) / N # identical
        #grad = np.sum((X @ theta.T - y) * X, axis=0) / N # identical
        
        theta = theta - alpha * grad
        cost[i] = computeCost(X, y, theta)
        if i < 5:
            print('fPrimeG: ', fPrimeG)
            print('g: ', g)
            print('grad: ', grad)
            print('theta: ', theta)
    
    return theta,cost

#set hyper parameters
alpha = 0.01
iters = 1000

t,cost = gradientDescent(X,y,theta,iters,alpha)
print('\nAfter {} iterations'.format(iters))
print('Theta vector: ', t)

finalCost = computeCost(X,y,t)
print('Final cost: ', finalCost)
