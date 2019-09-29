'''
sigmoid function sigma(z) = 1 / (1 + exp(-z))
'''
'''
**gradient descent: w = w + alpha * gradient(w)**
1. initialize the weight to be 1
2. repeat k times:
    calculate the gradient of all the data set
    update the w according to the equation above
    return the weight
'''
import numpy as np
import math

class LogisticRegression:
    def __init__(self, dataMat, dataLabel):
        self.data = dataMat
        self.labels = dataLabel
    
    def sigmoid(self, x):
        return 1.0 / (1 + math.exp(-x))
    
    def gradDescent(self, alpha=0.01, maxIter=1000): 
        '''
        apply the gradient descent/ascent algorithm
        '''
        data = self.data
        label = self.labels
        _, n = data.shape
        alpha = alpha                # set up a learning rate
        maxIter = maxIter            # set up a max update steps
        weight = np.ones((n, 1))     # initialize all weight to be one
        for _ in range(maxIter):
            h = self.sigmoid(np.dot(data, weight))
            err = label - h
            weight += alpha * np.dot(data.T, err)
        return weight


    def stocGradDescent(self, alpha=0.01):
        '''
        instead of calculate the gradient of all the data set
        we calucate the gradient of each sample
        '''
        data = self.data
        label = self.labels
        m, n = data.shape
        weights = np.ones((n, 1))
        for i in range(m):                                               # calculate the gradient of each sample
            h = self.sigmoid(np.sum(np.dot(data[i,:], weights)))  
            err = label[i] - h
            weights += alpha * np.dot(err, data[i,:])

        return weights

    
    def optStocGradDescent(self, maxIter=1000):
        '''
        1. descrease alpha to reduce the step interval
        2. random choose the sample
        '''
        data = self.data
        label = self.labels
        m, n = data.shape
        weights = np.ones((n, 1))
        for j in range(maxIter):
            for i in range(m):
                alpha = 4 /(1.0 + j + i) + 0.01
                randIdx = int(np.random.uniform(0, range(m)))       # randomly pick a sample
                h = self.sigmoid(np.sum(np.dot(data[randIdx, :], weights)))
                err = label[randIdx] - h
                weights += alpha * np.dot(err, data[randIdx])
                del (data[randIdx, :])                              # remove the chosen sample
        return weights