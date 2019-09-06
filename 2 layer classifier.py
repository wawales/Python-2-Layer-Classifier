#!/usr/bin/env python
# coding: utf-8

# In[12]:


import numpy as np
import sklearn.datasets
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
##########train_data_x,train_label_y###########
np.random.seed(0)
m = 500
X, y = sklearn.datasets.make_moons(m, noise=0.20)
# plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)
# y = y.reshape((200, 1))
# x1 = np.random.randn(2,100) + 1
# x2 = np.random.randn(2,100) + 5
# x = np.hstack((x1,x2))
# y1 = np.ones((1,100))
# y2 = np.zeros((1,100))
# y = np.hstack((y1,y2))
##########w,b###########
w1 = np.random.randn(2,10) / np.sqrt(2)
b1 = np.zeros((1,10))
w2 = np.random.randn(10,1) / np.sqrt(10)
b2 = np.zeros((1,1))
lr = 1
def f(x):
    x[x<=0] = 0
    x[x>0] = 1
    return x
for i in range(5000):
    ##########forwaord###########
    z1 = X.dot(w1) + b1
#     a1 = np.tanh(z1)
    a1 = np.maximum(0, z1)
    z2 = a1.dot(w2) + b2
    a2 = 1 / (1 + np.exp(-z2))
    J = - (1 * y.T.dot(np.log(a2)) + 1 * (1 - y.T).dot(np.log(1 - a2)))/m
    if i % 1000 == 0:
        print("epoch %d loss:%f" % (i,J))
    ##########backward############
    dz2 = -y.reshape((m,1)) + a2
    dw2 = a1.T.dot(dz2)/m
    db2 = np.sum(dz2, axis=0, keepdims=True)/m
    da1 = dz2.dot(w2.T)
#     dz1 = da1 * (1 - np.power(a1, 2))
    dz1 = da1 * f(z1)
    dw1 = X.T.dot(dz1)/m
    db1 = np.sum(dz1, axis=0, keepdims=True)/m
    ############optimer###########
    w1 += - lr * dw1
    b1 += - lr * db1
    w2 += - lr * dw2
    b2 += - lr * db2
def model(x):
    z1 = x.dot(w1) + b1
#     a1 = np.tanh(z1)
    a1 = np.maximum(0, z1)
    z2 = a1.dot(w2) + b2
    a2 = 1 / (1 + np.exp(-z2))
    a2[a2 <=0.5] = 0
    a2[a2 >0.5] = 1
    return a2
def plot_decision_boundary(pred_func):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
plot_decision_boundary(lambda x: model(x))
# plt.scatter(x1[0], x1[1])
# plt.scatter(x2[0], x2[1])
# plt.xlabel('x1')
# plt.ylabel('x2')
# plt.axis([-5, 10, -5, 10])
# plt.show()

