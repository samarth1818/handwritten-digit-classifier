import math
import heapq
import numpy as np


class KNNClassifier(object):
    def __init__(self, train, target, K):
        self.K = K
        self.trainFeatures = []
        for i in xrange(len(train)):
            arr = np.array(train[i], dtype=int)
            self.trainFeatures.append(arr)
        self.trainTargets = target

    def classify(self, X):

        # convert to numpy array
        npX = np.array(X, dtype=int)

        # define max-heap of k nearest neighbours
        neigh = [] # neighbor set of X

        for i in xrange(len(self.trainFeatures)):
            currFeat = self.trainFeatures[i]

            # calculate euclidean distance squared
            d = np.sum(np.power(np.subtract(npX, currFeat), 2))

            if len(neigh) < self.K:
                item = HeapItem(index=i, value=d)
                heapq.heappush(neigh, item)
            else:
                maxDist = neigh[0]
                if maxDist.value > d:
                    heapq.heappop(neigh)
                    item = HeapItem(index=i, value=d)
                    heapq.heappush(neigh, item)

        # count the occurences of each class in the neighbour set
        classoccurences = [0 for x in xrange(10)]
        for item in neigh:
            i = int(self.trainTargets[item.index])
            classoccurences[i] += 1

        # take class with maximum occurrences in the neighbour set
        maxOccurence = classoccurences[0]
        maxIndex = 0
        for x in xrange(10):
            if classoccurences[x] > maxOccurence:
                maxOccurence = classoccurences[x]
                maxIndex = x

        return maxIndex

    def test(self, inpSet, targetSet):
        # test this classification of the given input set and calculate the accuracy against the target set
        # returns accuracy of classification

        countA = 0.0    # record the correct classifications
        totalInputs = len(inpSet)
        j = int(totalInputs / 10)
        k = j
        for i in xrange(totalInputs):
            res = self.classify(inpSet[i])
            y = int(targetSet[i])
            if res == y:
                countA += 1.0
            if (i == k):
                print '#',
                k += j
        print " done"
        return countA / float(len(inpSet))


class HeapItem(object):
    def __init__(self, index, value):
        self.index = index
        self.value = value

    def __lt__(self, other):
        return self.value > other.value
