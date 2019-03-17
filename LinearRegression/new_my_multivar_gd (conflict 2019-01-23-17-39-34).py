import numpy as np
import pandas as pd

#data = pd.read_csv('simple3vardata.txt', names=['x1', 'x2', 'y'])

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
#print('costs\n', costs)

def cost(X, theta, y):
    return np.sum((np.dot(X, theta) - y) ** 2)

for i in range(epoch):
    g = np.dot(np.dot(X, theta) - y, X) / N
    theta = theta - alpha * g
    if i < 5:
        print('g: ', g)
        print('theta: ', theta)

print('After {} iterations'.format(i + 1))
print('g: ', g)
print('theta: ', theta)
print('cost: ', cost(X, theta, y))