'''
1. calculate the  distance from the label point to the current data point
2. sort the distance in increasing order
3. pick the first k data point 
4. count the frequency of label in the first k data point
5. assign the label with high frequency to the current data point
'''

import numpy as np
import collections

class KNN:
    def __init__(self, input, data, labels, k):
        self.input = input
        self.data = data
        self.labels = labels
        self.k = k

    def knn(self):
        data = self.data
        input = self.input
        k = self.k
        labels = self.labels
        m, n = data.shape
        diff = np.tile(input, (m, 1)) - data       # expand the input data to a match the labelled data
        dist = np.sqrt(np.sum(diff*diff, axis=1))  # so we can calcuate the distance based on a matrix.
        Idx = np.argsort(dist)
        classCounter = collections.defaultdict(int)
        for i in range(k):
            label = labels[Idx[i]]
            classCounter[label] += 1
        '''sort based on the value (frequency) in decreasing order'''
        sortedCount = sorted(classCounter.iteritems(), key=lambda x:x[1], reverse=True)
        
        '''assign the label(dict[0][0]) with the most frequency (dict[0]) to the input data'''
        return sortedCount[0][0]

    