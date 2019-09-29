


import math
import numpy as np
import collections

def calcShannonEnt(data):
    '''
    use Shannon Entropy to calculate the information gain 
    Entropy = - \sum^{n}_{i=1}p(x_{i})*log(p(x_{i}))
    '''
    numOfEntries = len(data)
    labelCounts = collections.defaultdict(int)
    for item in numOfEntries:
        currenLabel = item[-1]                  # the last column is the label
        labelCounts[currenLabel] += 1           # count the frequency of each label
    entropy = 0                                 # initialize the entropy as 0
    for key in labelCounts:
        prob = labelCounts[key]/float(numOfEntries)   # calculate the probablity for each label
        entropy -= prob * math.log(prob, 2)

    return entropy


def splitDataSet(data, axis, value):
    '''
    split the data
    '''
    temp = []
    for item in data:
        if item[axis] == value:
            featureVec = item[:axis]
            featureVec.extend(item[axis+1:])
            temp.append(featureVec)
    return temp

def chooseBestFeatureToSplit(data):
    '''
    before every branch, we choose the best feature based on the Shannon Entropy
    '''
    numofFeature = len(data) - 1
    baseEntropy = calcShannonEnt(data)
    bestIG = 0.0
    bestFeature = -1
    for i in range(numofFeature):
        feaList = [item[i] for item in data]
        uniVal = set(feaList)                               # create the unique label
        newEntropy = 0.0
        for value in uniVal:
            subDataSet = splitDataSet(data, i, value)       # calculate the entropy for each split
            prob = len(subDataSet)/float(len(data))
            newEntropy += prob * calcShannonEnt(subDataSet) 
        IG = baseEntropy - newEntropy
        if IG > bestIG:                                     # get the best information gain
            bestIG = IG
            bestFeature = i
    return bestFeature

def maxCount(classList):
    '''
    get the most frequent label
    '''
    classCount = collections.defaultdict(int)
    for vote in classList:
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=lambda x:x[1], reverse=True)
    '''return the most frequent feature'''
    return sortedClassCount[0][0]

def createTree(data, labels):
    classList = [sample[-1] for sample in data]

    '''if the classes are all the same, return'''
    if classList.count(classList[0]) == len(classList):
        return classList[0]

    '''traverse all the feature and return the most frequent one'''
    if len(data[0]) == 1:
        return maxCount(classList)

    bestFeature = chooseBestFeatureToSplit(data)
    bestFeatureLabel = labels[bestFeature]
    myTree = {bestFeatureLabel:{}}                      # use HashMap for the tree structure
    del(labels[bestFeature])                            # delete the feature that already used

    featureList = [sample[bestFeature] for sample in data]    # get all the features in the list
    uniVal = set(featureList)
    for val in uniVal:
        subLabels = labels[:]
        myTree[bestFeature][val] = createTree(splitDataSet(data, bestFeature, val), subLabels)
    
    return myTree
    
