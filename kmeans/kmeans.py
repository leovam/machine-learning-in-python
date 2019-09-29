
import numpy as np
class Kmean:
    def __init__(self, data, k):
        self.data = data
        self.k = k                 # k-cluster
        self.m = data.shape[0]     # number of samples
        self.n = data.shape[1]     # number of features
    
    def EuclidDist(self, arrA, arrB):
        '''calculate the Euclidean distance between two verctor'''
        return np.sqrt(np.sum([item*item for item in (arrA-arrB)]))
    
    def InitCenter(self):
        ''' randomly initialize a point as the center '''
        data = self.data
        n = self.n
        k = self.k
        centers = np.zeros((k, n))
        for i in range(n):
            minValue = np.min([data[:, i]])         # the mininum value of
            rangeI = np.max(data[:, i]) - minValue  # the range of the column
            centers[:, i] = minValue + rangeI * np.random.rand(k, 1) # make sure the center is within the data points
        return centers

    def kmeans(self):
        m = self.m
        k = self.k
        data = self.data
        cluster = np.zeros((m, 2))
        Centers = self.InitCenter()
        Changed = True                            # flag to continue the update of center or not
        while Changed:
            Changed = False
            for i in range(m):
                minDist, minIndex = np.inf, -1
                for j in range(k):
                    dist = self.EuclidDist(Centers[j, :], data[i, :])  # calculate the distance of each point
                    if dist < minDist:                                 # find the minimum distance
                        minDist = dist  
                        minIndex = j                                   # assign the label to the point
                if cluster[i, 0] != minIndex:
                    Changed = True
                cluster[i,:] = minIndex, minDist * minDist
        
            for j in range(k):
                pointInCluster = data[np.nonzero(cluster[:,0] == j)[0]]
                Centers[j, :] = np.mean(pointInCluster, axis=0)
        return Centers, cluster

