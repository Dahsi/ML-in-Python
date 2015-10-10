# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 11:56:38 2015

@author: alanchc
"""
from numpy import *
import operator
from os import listdir

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    #--Distance calculation START--
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    #--Distance calculation OVER--
    sortedDistIndicies = distances.argsort()
    classCount={}
    #--Voting w/ lowest k distances START--
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0)+1
    #--Voting w/ lowest k distances OVER--
    #--Sort dictionary START--
    sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    #--Sort dictionary OVER--
    return sortedClassCount[0][0]

# kNN.classify0([0,0], group, labels, 3)

def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())
    returnMat = zeros((numberOfLines,3))
    classLabelVector = []
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        #--Parse line to a list START--     
        line = line.strip()
        listFromLine = line.split('\t')
        #--Parse line to a list OVER--
        returnMat[index,:]=listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector

# datingDataMat, datingLabels = kNN.file2matrix('datingTestSet2.txt')

def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m,1))
    normDataSet = normDataSet/tile(ranges, (m,1))
    return normDataSet, ranges, minVals

# normMat, ranges, minVals = kNN.autoNorm(datingDataMat)

def datingClassTest():
    hoRation = 0.1
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRation)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:], normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],5)
        print "The classifier came back with %d, the real answer is: %d" %(classifierResult, datingLabels[i])
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
    print "The total error rate is: %f" %(errorCount/float(numTestVecs))

# kNN.datingClassTest()

def classifyPerson():
    resultList = ['not at all','in small doses','in large doses']
    percentTats = float(raw_input("percentage of time spent playing video games? "))
    ffMiles = float(raw_input("frequent flier miles earned per year? "))
    iceCream = float(raw_input("liters of ice cream consumed per year? "))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr-minVals)/ranges, normMat, datingLabels,3)
    print "You will probably like this person: ",resultList[classifierResult-1]

# kNN.classifyPerson()

def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

# testvector = kNN.img2vector('testDigits/0_13.txt')

def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('trainingDigits/%s'%(fileNameStr))
    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s'%(fileNameStr))
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        if classifierResult != classNumStr:
            errorCount += 1
    print "\nthe total number of errors is: %d"% errorCount
    print "\nthe total error rate is: %f" % errorCount/float(mTest)

# kNN.handwritingClassTest()