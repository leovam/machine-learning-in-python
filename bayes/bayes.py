
import numpy as np
import math

def trainNB(trainMat, trainClass):
    numTrain = len(trainMat)
    numWords = len(trainMat[0])

    pAbusive = np.sum(trainClass)/float(numTrain)
    p0Num = np.ones(numWords)                   # avoid to multiply 0
    p1Num = np.ones(numWords)                   # if a value = 0, the result will be 0
    p0Denom = 2.0
    p1Denom = 2.0

    for i in range(numTrain):
        if trainClass[i] == 1:
            p1Num += trainMat[i]
            p1Denom += np.sum(trainMat[i])
        else:
            p0Num += trainMat[i]
            p0Denom += sum(trainMat[i])
    
    p1Vect = math.log(p1Num/p1Denom, 2)        # avoid the very small number to be 0
    p0Vect = math.log(p0Num/p0Denom, 2)

    return p0Vect, p1Vect, pAbusive
    

def classifierNB(vec, p0Vect, p1Vect, pClass):
    p1 = np.sum(vec * p1Vect) + math.log(pClass, 2)
    p0 = np.sum(vec * p0Vect) + math.log(1-pClass, 2)

    return 1 if p1 > p0 else 0