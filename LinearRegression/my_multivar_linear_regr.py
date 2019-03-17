# This version matches exactly the results of the Coursera Octave version !
import numpy as np
import pandas as pd

data = pd.read_csv('home.txt', names=['surface', 'rooms', 'price'])
print('data\n', data.head())
X = np.array(data.iloc[:,0:2])
print('X\n', X)
Xm = X - np.mean(X, axis=0)
print('Xm\n',Xm)
sigma = np.std(X, axis=0, ddof=1)
print('sigma\n', sigma)
X = np.divide(Xm, sigma)
print('normalized data\n', X)
#X = np.insert(X, 0, np.ones(X.shape[0]), axis=1) ok but more complicated
X = np.insert(X, 0, 1, axis=1)

print('X\n', X)
y = np.array(data.iloc[:,2])
print('y\n', y)
theta = np.zeros(X.shape[1])
print('theta\n', theta)

epoch = 400
alpha = 0.01
precision = 0.0001
N = X.shape[0]

costs = np.zeros(epoch)
print('costs\n', costs)
previousCost = 0

for i in range(epoch):
    yhat = X.dot(theta)
    error = yhat - y
    grad = error.dot(X) / N
    #grad = X.T.dot(yhat) / N
    theta = theta - alpha * grad
    cost = np.sum(error ** 2) / N
    costs[i] = cost
   
    if i < 5:
        print('yhat\n', yhat)
        print('grad\n', grad)         
        print('theta\n', theta)
        print('cost\n', cost)
        
    if abs(cost - previousCost) <= precision:
        break
    
    previousCost = cost  
    

print('theta\n', theta)
print('cost\n', cost)
print('i\n', i)
 