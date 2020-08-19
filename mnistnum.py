# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 11:46:52 2020

@author: manop
"""
#importing libraries
import gzip
import pickle
import numpy as np
import random
""" importing data """
with gzip.open('C:/Users/manop/Desktop/mnist.pkl.gz','rb') as f:
    u=pickle._Unpickler(f)
    u.encoding='latin1' 
    tr_data, va_data, te_data=u.load()
f.close() 
def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
training_inputs = [np.reshape(x, (784, 1)) for x in tr_data[0]]
training_results = [vectorized_result(y) for y in tr_data[1]]
training_data = list(zip(training_inputs, training_results))
validation_inputs = [np.reshape(x, (784, 1)) for x in va_data[0]]
validation_data = list(zip(validation_inputs, va_data[1]))
test_inputs = [np.reshape(x, (784, 1)) for x in te_data[0]]
test_data = list(zip(test_inputs, te_data[1]))

""" creating a neural netowrk """
#functions required for this code
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

def ff(a):
    for b,w in zip(biases,weights):
        a = sigmoid(np.dot(w,a)+b)
    return a
def estimate(test_data):
    test_results = [(np.argmax(ff(x)), y) for (x, y) in test_data]
    return sum(int(x == y) for (x, y) in test_results)
#actual code starts here
sizes = [784,35,10]
num_layers = len(sizes)
biases = [np.random.randn(y, 1) for y in sizes[1:]]
weights = [np.random.randn(y, x)for x, y in (sizes[:-1], sizes[1:])]
epochs = 15
mb_size = 10
eta = 3.0
if test_data: n_test = len(test_data)
n = len(training_data)
for j in range(epochs):
    random.shuffle(training_data)
    mb = [training_data[k:k+mb_size] for k in range(0, n, mb_size)]
    for minibatch in mb:
        b_lbl = [np.zeros(b.shape) for b in biases]
        w_lbl = [np.zeros(w.shape) for w in weights]
        for x,y in minibatch:
            db_lbl = [np.zeros(b.shape) for b in biases]
            dw_lbl = [np.zeros(w.shape) for w in weights]
            activation = x
            activations = [x]
            zs = []
            for b,w in zip(biases,weights):
                z = np.dot(w,activation)+b
                zs.append(z)
                activation = sigmoid(z)
                activations.append(activation)    
            delta = (activations[-1]-y)*sigmoid_prime(zs[-1])
            db_lbl[-1] = delta
            dw_lbl[-1] = np.dot(delta,np.transpose(activations[-2]))
            for l in range(2,num_layers):
                z = zs[-l]
                sp = sigmoid_prime(z)
                delta = np.dot(np.transpose(weights[-l+1]), delta) * sp
                db_lbl[-l] = delta
                dw_lbl[-l] = np.dot(delta,np.transpose(activations[-l-1]))
            b_lbl = [nb+dnb for nb,dnb in zip(b_lbl,db_lbl)]
            w_lbl = [nw+dnw for nw,dnw in zip(w_lbl,dw_lbl)]
        biases = [b -(eta/len(minibatch))*bl for b,bl in zip(biases,b_lbl)]
        weights = [w -(eta/len(minibatch))*wl for w,wl in zip(weights,w_lbl)]
    if test_data:
        print("Epoch {0}: {1} / {2}".format(j, estimate(test_data), n_test))
        











          