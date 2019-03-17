import numpy as np
import pandas as pd

data = pd.read_csv('home.txt', names=['surface', 'rooms', 'price'])
print('data\n', data.head())
data = (data - data.mean()) / data.std()
print('normalized data\n', data)
X = np.array(data.iloc[:,0:2])
print('X\n', X)
#X = np.insert(X, 0, np.ones(X.shape[0]), axis=1) ok but more complicated
X = np.insert(X, 0, 1, axis=1)

print('X\n', X)
y = np.array(data.iloc[:,2])
print('y\n', y)
theta = np.zeros(X.shape[1])
print('theta\n', theta)

epoch = 1000
alpha = 0.01
N = X.shape[0]

costs = np.zeros(epoch)
print('costs\n', costs)

for i in range(epoch):
    yhat = X.dot(theta) - y
    grad = yhat.dot(X) / N
    #grad = X.T.dot(yhat) / N
    theta = theta - alpha * grad
    cost = np.sum(yhat ** 2) / N
    costs[i] = cost
   
    if i < 5:
        print('yhat\n', yhat)
        print('grad\n', grad)         
        print('theta\n', theta)
        print('cost\n', cost)

print('theta\n', theta)
print('cost\n', cost)
 