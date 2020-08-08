# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 02:14:13 2020

@author: manop
"""

import numpy as np 

x = np.array([[0, 0, 1, 1],[0, 1, 0, 1]])  
y = np.array([[0, 1, 1, 0]]) 

def sigmoid(z): 
    return 1 / (1 + np.exp(-z)) 

#w1 is the weights for the first hidden layer.. w2 is the weights for the output layer
w1 = np.random.randn(2,2) 
w2 = np.random.randn(1,2)

#b1 is the bias for the first hidden layer.. and b2 is bias for the output layer  
b1 = np.ones((2, 1)) 
b2 = np.ones((1, 1))

z1 = np.dot(w1,x)+b1
p1 = sigmoid(z1)

z2 = np.dot(w2,p1)+b2
ye = sigmoid(z2)

# updating weights

for n in range(200000):
    dz2 = ye - y
    dw2 = np.dot(dz2,p1.T)/x.shape[1]
    db2 = np.sum(dz2,axis=1,keepdims=True)/x.shape[1]
    dp1 = np.dot(w2.T,dz2)
    dz1 = np.multiply(dp1,p1*(1-p1))
    dw1 = np.dot(dz1,x.T)/x.shape[1]
    db1 = np.sum(dz1,axis=1,keepdims=True)/x.shape[1]
    
    w2 = w2 - 0.01*dw2
    b2 = b2 - 0.01*db2
    w1 = w1 - 0.01*dw1
    b1 = b1 - 0.01*db1
    z1 = np.dot(w1,x)+b1
    p1 = sigmoid(z1)
    z2 = np.dot(w2,p1)+b2
    ye = sigmoid(z2)

output = (ye>0.5)*1.0    
print(output)  