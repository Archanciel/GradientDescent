import numpy as np
import pandas as pd

A = np.array([[1,2,3],[10,20,30],[100,200,300],[1000,2000,3000]])
print('A\n', A)
B = np.array([[1,2,3,4],[10,20,30,40],[100,200,300,400]])
print('B\n', B)
print('B x A\n', np.dot(B, A))
print('At x Bt\n', np.dot(A.T, B.T))
print('(At x Bt)t\n', np.dot(A.T, B.T).T)
print('B x A == (At x Bt)t\n', np.dot(B, A) == np.dot(A.T, B.T).T)