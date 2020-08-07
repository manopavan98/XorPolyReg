# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 17:19:56 2020

@author: manopavan
"""
import numpy as np
train = np.loadtxt(fname = "USPS_train-1.txt")
test = np.loadtxt(fname = "USPS_test-1.txt")
xtrain = train[:,0:16]
ytrain = train[:,16]
xtest = test[:,0:16]
ytest = test[:,16]
k= int(input("enter k value: "))
meantrain = np.mean(xtrain,axis=0)
stdtrain = np.std(xtrain,axis=0)
normtrain = np.divide((xtrain - meantrain),stdtrain)
meantest = np.mean(xtest,axis=0)
stdtest = np.std(xtest,axis=0)
normtest = np.divide((xtest - meantest),stdtest)
[rxtrain,cxtrain] = np.shape(xtrain)
[rxtest,cxtest] = np.shape(xtest)
labels=[]
for m in range(0,rxtest):
    distance=[]
    for p in range(0,rxtrain):
        dist=np.linalg.norm(normtest[m]-normtrain[p])
        distance.append((ytrain[p],dist))
    distance.sort(key=lambda tup: tup[1])  
    a=distance[:k]
    output=[row[0] for row in a]
    predict=max(set(output), key=output.count)
    labels.append(predict)
miss = ytest - labels
missn = np.where(miss==0)[0]
mce = np.shape(missn)[0]
accuracy=(mce/rxtest)*100
print("accuracy = "+str(accuracy))


    
    
       
        
    




 
