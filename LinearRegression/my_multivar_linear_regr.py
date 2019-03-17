# This version matches exactly the results of the Coursera Octave version !
import numpy as np
import pandas as pd

verbose = False

data = pd.read_csv('home.txt', names=['surface', 'rooms', 'price'])

if verbose:
    print('data\n', data.head())

X = np.array(data.iloc[:,0:2])
X_before_norm = X

if verbose:
    print('X\n', X)

Xm = X - np.mean(X, axis=0)

if verbose:
    print('Xm\n',Xm)

sigma = np.std(X, axis=0, ddof=1)

if verbose:
    print('sigma\n', sigma)

X = np.divide(Xm, sigma)

if verbose:
    print('normalized data\n', X)

#X = np.insert(X, 0, np.ones(X.shape[0]), axis=1) ok but more complicated
X = np.insert(X, 0, 1, axis=1)

if verbose:
    print('X\n', X)

y = np.array(data.iloc[:,2])

if verbose:
    print('y\n', y)

theta = np.zeros(X.shape[1])

if verbose:
    print('theta\n', theta)

epoch = 400
alpha = 0.1
precision = 0.0001
N = X.shape[0]

costs = np.zeros(epoch)

if verbose:
    print('costs\n', costs)

previousCost = 0

for i in range(epoch):
    yhat = X.dot(theta)
    error = yhat - y
    grad = error.dot(X) / N
    #grad = X.T.dot(yhat) / N
    theta = theta - alpha * grad
    cost = np.sum(error ** 2) / (2 * N)
    costs[i] = cost
   
    if verbose and i < 5:
        print('yhat\n', yhat)
        print('grad\n', grad)         
        print('theta\n', theta)
        print('cost\n', cost)
        
    if abs(cost - previousCost) <= precision:
        break
    
    previousCost = cost  
    
print('alpha\n', alpha)
print('theta\n', theta)
print('cost\n', cost)
print('i\n', i)

X_before_norm = np.insert(X_before_norm, 0, 1, axis=1)
#print('X_before_norm\n', X_before_norm)

X = X_before_norm

print('y\n', y)
theta_normal_equation = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)),X.T), y)
print('theta_normal_equation\n', theta_normal_equation)
print('Prediction for 1650, 3\n',np.dot(np.array([1,1650,3]), theta_normal_equation))
