# -*- coding: utf-8 -*-
"""
Created on Fri Oct 09 09:23:01 2015

@author: Dahsi
"""
from numpy import *
import random

def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('dataSet2.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1]), float(lineArr[2]), float(lineArr[3])])
        labelMat.append(int(lineArr[4]))
    fr.close()
    return dataMat, labelMat

def loadTestSet():
    dataMat = []
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1]), float(lineArr[2]), float(lineArr[3])])
        labelMat.append(int(lineArr[4]))
    fr.close()
    return dataMat, labelMat

def main():
    X, Y = loadDataSet()
    TX, TY = loadTestSet()
    pocket(X, Y, TX, TY, 1000)

def sign(x):
    if x <= 0:
        return -1
    return 1

# -- Pocket Alg. START --
def errorCount(X, Y, indList, weights):
    error = 0    
    for index in indList:
        if(sign(inner(weights,X[index])) != Y[index]):
            error += 1
    return error

def paTrain(X, Y, wp=True):
    nY = len(Y)
    nFeature = len(X[0])
    weights = zeros(nFeature)
    wPocket = weights
    indList = random.sample(range(nY),nY)
    minError = errorCount(X, Y, indList, weights)
    numUpdates = 0
    for index in indList:
        if(sign(inner(weights,X[index])) != Y[index]):
            weights = weights + Y[index] * array(X[index])
            nError = errorCount(X, Y, indList, weights)
            numUpdates += 1
            if nError < minError:
                minError = nError
                wPocket = weights
        if numUpdates == 50:
            break
    if wp == True:
        return wPocket
    else:
        return weights

def pocket(X, Y, TX, TY, nRepeat):
    sumAveError = 0
    for i in range(nRepeat):
        print "i = ",i
        #weights = paTrain(X, Y)
        weights = paTrain(X, Y, False)
        indList = range(len(TY))
        testError = errorCount(TX, TY, indList, weights)
        aveError = float(testError)/float(len(TY))
        sumAveError += aveError
    print sumAveError/float(nRepeat)
# -- Pocket Alg. OVER --