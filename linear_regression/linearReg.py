'''
square error = (y-x^T*w)^2
take the derivative of w and set to 0:
    X^T(Y-Xw) = 0
we have w_hat = (X^T X)^{-1} X^T y
'''

import numpy as np
import math


def standRegres(xMat, yMat):
    '''
    treat all the point evenly, there is no-bias classify
    '''
    xTx = np.dot(xMat.T, xMat)
    if np.linalg.det(xTx) == 0:       # if teh determinant is 0, then there is no inverse
        return 
    
    ws = np.linalg.solve(xTx, np.dot(xMat.T, yMat.T))
    return ws

def lwlrRegress(test, xMat, yMat, k =1.0):
    '''
    give a weight to the neighbouring point
    w_hat = (X^T W X)^{-1} X^T W y
    use Gaussian kernel as the weight:
    w(i,i) = exp(|x_i-x|/(-2k^2))
    '''
    m = xMat.shape[0]
    weight = np.eye((m))                    # diagnal matrix of the weight
    for j in range(m):                      # assign the weight value to the diagonal
        diff = test - xMat[j, :]
        weight[j, j] = math.exp(np.dot(diff.T, diff)/(-2.0 * k * k))
    xTx = np.dot(xMat.T, np.dot(weight, xMat))
    if np.linalg.det(xTx) == 0:
        return 
    ws = np.dot(xTx.T, np.dot(xMat.T, np.dot(weight, yMat)))
    return np.dot(test, ws)


'''
the two method works only with full rank matrix
however, if the number of features > number of sample, they do not work
then we need introduce penalty
'''

'''
ridge regression:
w_hat = (X^T X + lambda I)^{-1} X^T y
'''
def ridgeReg(xMat, yMat, lam=0.2):
    xTx = np.dot(xMat.T, xMat)
    de = xTx + np.eye(xMat.shape[1]) * lam

    if np.linalg.det(de) == 0:
        return
    ws = np.dot(de.I, np.dot(xMat.T, yMat))

    return ws

'''
stageWise: greedy aglorithm
reduce the error at each step
'''
def rssError(yMat, yMatHat):
    return ((yMat-yMatHat)**2).sum()

def regularize(xMat):                    # regularize by column
    inMat = xMat.copy()
    inMean = np.mean(inMat, axis=0)      # substract mean from the value
    inVar = np.var(inMat, axis=0)        # calculate the variance
    inMat = (inMat - inMean) / inVar
    return inMat

def stageWise(xMat, yMat, eps=0.01, maxIter=100):
    yMat = yMat - np.mean(yMat, axis=0)
    xMat = regularize(xMat)
    m, n = xMat.shape
    returnMat = np.zeros((maxIter, n))
    ws = np.zeros((n,1))
    wsTest = ws.copy()
    wsMax = ws.copy()
    for i in range(maxIter):
        lowestError = np.inf
        for j in range(n):
            for sign in [-1, 1]:
                wsTest = ws.copy()
                wsTest[j] += eps * sign
                yTest = xMat * wsTest
                rssE = rssError(yMat, yTest)
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i,:] = ws.T

    return returnMat
