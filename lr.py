# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 23:19:52 2020

@author: manop
"""
"importing numpy library"
import numpy as np;

"loading the data sets"
tri = np.loadtxt(fname = "Tri.txt")
tro = np.loadtxt(fname = "Tro.txt")

"concatenating ones with input matrix"
x = np.hstack((np.ones((tri.shape[0],1)),tri))
y = tro.reshape([tro.shape[0],1])    
y=y.T    

"creating weight matrix with random values"
w = np.random.rand(1,4)

"finding out the predicting output"
ye = np.dot(w,x.T)

"training the model"
for n in range(100):
    w = w - 0.02*np.dot((ye-y),x)
    ye = np.dot(w,x.T)
print(w)
    
"finding residual sum of squares"        
e = y-ye
e1 = np.transpose(y-ye)
error = np.dot(e,e1)
"error is 0.86..."
print(error)    
 

    
#


