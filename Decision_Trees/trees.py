# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 11:27:18 2015

@author: Dahsi
"""

from math import log
import operator

def createDataSet():
    """
    dataSet = [["<=30", "高", "N", "O", "N"],  
              ["<=30", "高", "N", "G", "N"],  
              ["31-40", "高", "N", "O", "Y"],  
              [">40", "中", "N", "O", "Y"],  
              [">40", "低", "Y", "O", "Y"],  
              [">40", "低", "Y", "G", "N"],  
              ["31-40", "低", "Y", "G", "Y"],  
              ["<=30", "中", "N", "O", "N"],  
              ["<=30", "低", "Y", "O", "Y"],  
              [">40", "中", "Y", "O", "Y"],  
              ["<=30", "中", "Y", "G", "Y"],  
              ["31-40", "中", "N", "G", "Y"],  
              ["31-40", "高", "Y", "O", "Y"],  
              [">40", "中", "N", "G", "N"]]  
    labels = ["Age", "Income", "Student?", "CC", "Class"]
    return dataSet, labels
    """   
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']

    return dataSet, labels

myDat, labels = createDataSet()


def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    # --Create dictionary of all possible classes START--
    for featVect in dataSet:
        currentLabel = featVect[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    # --Create dictionary of all possible classes OVER--
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob*log(prob, 2)
    return shannonEnt

# trees.calcShannonEnt(myDat)


def splitDataSet(dataSet, axis, value):
    # --Create separate list START--
    retDataSet = []
    # --Create separate list OVER--
    for featVec in dataSet:
        if featVec[axis] == value:
            # --Cut out the feature split on START--
            reducedFeatVec = featVec[:axis]  # add elements before featVec[axis]
            reducedFeatVec.extend(featVec[axis+1:])  # add elements after featVec[axis]
            retDataSet.append(reducedFeatVec)
            # --Cut out the feature split on OVER--
    return retDataSet

# trees.splitDataSet(myDat,0,1)


def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0])-1  # because last col. is class label
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        # --Create unique list of class labels START--
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        # --Create unique list of class labels OVER--
        newEntropy = 0.0
        # --Calculate entropy for each split START--
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        # --Calculate entropy for each split OVER--
        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

# trees.chooseBestFeatureToSplit(myDat)


def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    # --Stop when all classes are equal START--
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # --Stop when all classes are equal OVER--
    # --When no more features, return majority START--
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    # --When no more features, return majority OVER--
    bestFeat = chooseBestFeatureToSplit(dataSet)
    tempLabels = labels[:]  # copy labels list to tempLabels in order to aviod affecting labels list
    bestFeatLabel = tempLabels[bestFeat]
    myTree = {bestFeatLabel: {}}
    del(tempLabels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = tempLabels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree

# myTree = trees.createTree(myDat, labels)


def classify(inputTree, featLabels, testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel

# trees.classify(myTree, labels, [1,0])


def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, 'w')
    pickle.dump(inputTree, fw)
    fw.close()


def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)

# trees.storeTree(myTree, 'classifierStorage.txt')
# trees.grabTree('classifierStorage.txt')

"""
fr = open('lenses.txt')
lenses = [inst.strip().split('\t') for inst in fr.readlines()]
lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
lensesTree = trees.createTree(lenses, lensesLabels)
"""
