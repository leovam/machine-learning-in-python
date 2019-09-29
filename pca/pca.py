'''
1. remove the mean
2. calculate the covariance matrix M
3. calculate the eigenvalue and eigenvector of M
4. sorted the eigenvalue in increasing order
5. keep the largest N eigenvector
6. transform the N eigenvector to new space
'''


import numpy as np

class PCA:
    def __init__(self, data):
        '''row * column = sample * features'''
        self.data = data
    
    def pca(self, topFeatureNum=10000000):
        mean = np.mean(self.data, axis=0)               # calculate the feature mean in each sample 
        meanRemoved = self.data - mean                  # remove the mean from the original data
        covMat = np.cov(meanRemoved, rowvar=False)      # calculate covariance, column is variable
        eigValues, eigVectors = np.linalg.eig(covMat)   # calculate the eigenValues and eigenVectors
        eigValuesIndex = np.argsort(eigValues)          # get the index of sorted array
        eigValues = eigValues[:-(topFeatureNum+1):-1]   # get the top N features from min to max
        sortedEigVectors = eigVectors[:, eigValuesIndex]
        lowData = meanRemoved * sortedEigVectors        # project the data to the new space
        reconMat = (lowData * sortedEigVectors.T) + mean
        return lowData, reconMat
