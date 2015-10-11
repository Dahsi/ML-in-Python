# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 15:21:53 2015

@author: Dahsi
"""

from numpy import *
import random


def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('dataSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1]), float(lineArr[2]), float(lineArr[3])])
        labelMat.append(int(lineArr[4]))
    fr.close()
    return dataMat, labelMat

def main():
    X, Y = loadDataSet()  
    naiveCycle(X, Y)
    predef(X, Y, 1000)
    predef(X,Y,1000,0.5)
    
def sign(x):
    if x <= 0:
        return -1
    return 1


# -- Perceptron Learning Alg. START --

def plaTrain(X, Y, rand=False, alpha=1):
    nY = len(Y)
    nFeature = len(X[0])
    weights = zeros(nFeature)
    numUpdates = 0
    if rand == True:
        IndList = random.sample(range(nY),nY)
    else:
        IndList = range(nY)
    while 1:
        flag = 0
        for index in IndList:
            if(sign(inner(weights,X[index])) != Y[index]):
                weights = weights + Y[index] * array(X[index])
                numUpdates += 1
        # -- check all X again. If there is wrong classification, start classifying again.
        for index in IndList:
            if(sign(inner(weights,X[index])) != Y[index]):
                flag = 1
        if flag == 0:
            break
    return numUpdates

def naiveCycle(X, Y):
    numUpdates = plaTrain(X, Y)    
    print numUpdates
   
def predef(X, Y, nRepeat, alpha = 1):
    total = 0
    for i in range(nRepeat):
        numUpdates = plaTrain(X,Y, True, alpha)
        total += numUpdates 
    print float(total)/float(nRepeat)
  
# -- Perceptron Learning Alg. OVER --
