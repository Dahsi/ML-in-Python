# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 17:12:07 2015

@author: Dahsi
"""

from sklearn import neighbors
#import numpy as np
import numpy as np
from os import listdir

# -- turn 32X32 array into 1*1024 vector start --
def img2vector(filename):
    returnVect = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect
# -- turn 32X32 array into 1*1024 vector start --

# -- load training data start --
def loadDataSet(filename):
    hwLabels = []
    dataFileList = listdir(filename)
    m = len(dataFileList)
    DataArr = np.zeros((m,1024))
    for i in range(m):
        fileNameStr = dataFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        DataArr[i,:] = img2vector('%s/%s' % (filename, fileNameStr))
    return np.array(DataArr), np.array(hwLabels)
# -- load training data over --

# -- kNN classification start --  
def knnClassify(n_neighbors, weights, X_train, Y_train, X_test, Y_test):
    knn = neighbors.KNeighborsClassifier(n_neighbors, weights)
    knn.fit(X_train, Y_train)
    error = 1-knn.score(X_test, Y_test)
    return error
# -- kNN classification over --

if __name__ == "__main__":
    X_train, Y_train = loadDataSet('trainingDigits')
    X_test, Y_test = loadDataSet('testDigits')
    k = 3
    error = knnClassify(k, 'uniform', X_train, Y_train, X_test, Y_test)
    print "the total error rate is: ",error
    
    
    
