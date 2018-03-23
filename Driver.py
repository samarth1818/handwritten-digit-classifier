from sklearn.datasets import fetch_mldata
from Classifier import KNNClassifier
import random
import matplotlib.pyplot as plt

# fetch the MNIST original data set
mnist = fetch_mldata('MNIST original')
mnistInputSet = mnist.data[:]
mnistTargetSet = mnist.target[:]

# randomize this data
z = zip(mnistInputSet, mnistTargetSet)
random.shuffle(z)
mnistInputSet, mnistTargetSet = zip(*z)

# separate train and test data sets
totalTrain = 6000
totalTest = 1000
trainInputSet = mnistInputSet[:totalTrain]
trainTargetSet = mnistTargetSet[:totalTrain]
testInputSet = mnistInputSet[totalTrain:totalTrain+totalTest]
testTargetSet = mnistTargetSet[totalTrain:totalTrain+totalTest]

# conduct tests for different k values
kvalues = [1, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99]
trainErrorSet = [0.0 for x in xrange(11)]
testErrorSet = [0.0 for x in xrange(11)]

for i in xrange(len(kvalues)):
    K = kvalues[i]
    knn = KNNClassifier(trainInputSet, trainTargetSet, K)   # initialize classifier
    print ("Testing for k = " + str(K) + " on test set:")
    acc = knn.test(testInputSet, testTargetSet) # accuracy measure for test data
    err = 1.0 - acc # error for test data
    print ("Test Error = " + str(err))
    print ""
    testErrorSet[i] = err
    # print ("Testing for k = " + str(K) + " on train set:")
    # acc = knn.test(trainInputSet, trainTargetSet)   # accuracy measure for test data
    # err = 1.0 - acc # accuracy measure for test data
    # print ("Train Error = " + str(err))
    # print ""
    # print ""
    # trainErrorSet[i] = err


# plot
plt.plot(kvalues, testErrorSet, "-", kvalues, testErrorSet, "o")
plt.yscale('linear')
plt.show()
