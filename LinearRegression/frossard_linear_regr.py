

# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# *Load necessary libraries*

# In[2]:


data_x = np.linspace(1.0, 10.0, 100)[:, np.newaxis]
data_y = np.sin(data_x) + 0.1 * np.power(data_x, 2) + 0.5 * np.random.randn(100, 1)


# *Generate our data*

# In[3]:


model_order = 6
data_x = np.power(data_x, range(model_order))
data_x /= np.max(data_x, axis=0)


# *Add intercept data and normalize*

# In[4]:


order = np.random.permutation(len(data_x))
portion = 20
test_x = data_x[order[:portion]]
test_y = data_y[order[:portion]]
train_x = data_x[order[portion:]]
train_y = data_y[order[portion:]]


# *Shuffle data and produce train and test sets*

# In[5]:


def get_gradient(w, x, y):
    y_estimate = x.dot(w).flatten()
    error = (y.flatten() - y_estimate)
    mse = (1.0/len(x))*np.sum(np.power(error, 2))
    gradient = -(1.0/len(x)) * error.dot(x)
    return gradient, mse


# *Create gradient function*

# In[6]:


w = np.random.randn(model_order)
alpha = 0.5
tolerance = 1e-5

# Perform Stochastic Gradient Descent
epochs = 1
decay = 0.99
batch_size = 10
iterations = 0
while True:
    order = np.random.permutation(len(train_x))
    train_x = train_x[order]
    train_y = train_y[order]
    b=0
    while b < len(train_x):
        tx = train_x[b : b+batch_size]
        ty = train_y[b : b+batch_size]
        gradient = get_gradient(w, tx, ty)[0]
        error = get_gradient(w, train_x, train_y)[1]
        w -= alpha * gradient
        iterations += 1
        b += batch_size
    
    # Keep track of our performance
    if epochs%100==0:
        new_error = get_gradient(w, train_x, train_y)[1]
        print("Epoch: {0} - Error: {1:1.4f}".format(epochs, new_error))
    
        # Stopping Condition
        if abs(new_error - error) < tolerance:
            print("Converged.")
            break
        
    alpha = alpha * (decay ** int(epochs/1000))
    epochs += 1

print("w =",w)
print("Test Cost =", get_gradient(w, test_x, test_y)[1])
print("Total iterations =", iterations)


# *Perform gradient descent to learn model*

# In[7]:


y_model = np.polyval(w[::-1], np.linspace(0,1,100))
plt.plot(np.linspace(0,1,100), y_model, c='g', label='Model')
plt.scatter(train_x[:,1], train_y, c='b', label='Train Set')
plt.scatter(test_x[:,1], test_y, c='r', label='Test Set')
plt.grid()
plt.legend(loc='best')
plt.xlabel('X')
plt.ylabel('Y')
plt.xlim(0,1)
plt.show()


# *Plot the model obtained*