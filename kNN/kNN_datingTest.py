# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 16:13:33 2015

@author: Dahsi
"""

from sklearn import neighbors, cross_validation
import numpy as np
import matplotlib.pyplot as plt


# -- load dataset start --
def loadDataset():
    fr = open("datingTestSet2.txt")
    dataArr = []
    labelArr = []
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataArr.append([float(lineArr[i]) for i in range(len(lineArr)-1)])
        labelArr.append(lineArr[-1])
    fr.close()
    return np.array(dataArr), np.array(labelArr)
# -- load dataset over --        
    
# -- create scatter plots start --
def createPlot(a):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(a[:,1],a[:,2])
    plt.show()
# -- create scatter plots over --

# -- normalize data start --    
def autoNorm(dataset):
    minVals = dataset.min(axis=0)
    maxVals = dataset.max(axis=0)
    ranges = maxVals - minVals
    normDataset = np.zeros(np.shape(dataset))
    m = dataset.shape[0]
    normDataset = dataset - tile(minVals, (m,1))
    normDataset = normDataset/tile(ranges, (m,1))
    return normDataset
# -- normalize data over --  

# -- kNN classification w/ (1/valRation)-fold CV start --  
def knnClassify(n_neighbors, weights, valRatio, normDataset, labelSet):
    knn = neighbors.KNeighborsClassifier(n_neighbors, weights)
    score = cross_validation.cross_val_score(knn, normDataset, labelSet, cv = int(1/valRatio))
    aveError = 1-score.mean()
    return aveError
# -- kNN classification w/ (1/valRation)-fold CV over --
 
if __name__ == "__main__":
    datingData, datinglabel = loadDataset() # load data
    k = 3
    valRatio = 0.1
    #createPlot(datingData) # create plot
    normDatingData = autoNorm(datingData) # normal feature given equal weight for all feature
    aveError = knnClassify(k, 'uniform', valRatio, normDatingData, datinglabel)
    print "the total error rate is ",aveError

