import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def loadData():
    with np.load('notMNIST.npz') as data :
        Data, Target = data ['images'], data['labels']
        posClass = 2
        negClass = 9
        dataIndx = (Target==posClass) + (Target==negClass)
        Data = Data[dataIndx]/255.
        Target = Target[dataIndx].reshape(-1, 1)
        Target[Target==posClass] = 1
        Target[Target==negClass] = 0
        np.random.seed(421)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data, Target = Data[randIndx], Target[randIndx]
        trainData, trainTarget = Data[:3500], Target[:3500]
        validData, validTarget = Data[3500:3600], Target[3500:3600]
        testData, testTarget = Data[3600:], Target[3600:]
    return trainData, validData, testData, trainTarget, validTarget, testTarget

# for training: W^T must be a 28x28 weight matrix
# # x[i] = 28x28 (picture)

def MSE(W,b,x,y,reg):
    numSamples = x.shape[0]
    numPixels = x.shape[1]*x.shape[2]
    W_ = W.reshape((numPixels, 1))
    X_ = x.reshape((numSamples, numPixels))
    WX = np.matmul(X_, W_)
    cost = WX + b - y
    cost = (np.linalg.norm(cost)**2)/(2*numSamples)
    regu = np.matmul(W_.transpose(), W_)*reg/2
    return np.sum(cost+regu)


def gradMSE(W, b, x, y, reg):
    #calculate gradient with respect to W
    numSamples = x.shape[0]
    numPixels = x.shape[1]*x.shape[2]
    W_ = W.reshape((numPixels, 1))
    X_ = x.reshape((numSamples, numPixels))
    c = np.matmul(X_, W_) + b - y
    gradW = np.matmul(X_.transpose(), c)/numSamples + reg*W_
    gradb = np.sum(c)/numSamples
    return gradW, gradb


def crossEntropyLoss(W, b, x, y, reg):
    # Your implementation here

def gradCE(W, b, x, y, reg):
    # Your implementation here

def grad_descent(W, b, trainingData, trainingLabels, alpha, iterations, reg, EPS):
    # Your implementation here
    error = 2*EPS
    W = W.reshape((784,1))
    i=0
    e_list = []
    while(error > EPS and i<iterations):
        #print(MSE(W,b,trainingData,trainingLabels,reg))
        e_list.append(MSE_vector(W,b,trainingData,trainingLabels,reg))
        gradW, gradb = gradMSE_vector(W,b,trainingData,trainingLabels,reg)
        error = np.linalg.norm(gradW)
        W = np.subtract(W, alpha*gradW)
        b = b - alpha*gradb
        #error = MSE(W,b,trainingData,trainingLabels,reg)
        i+=1

    return W, b, e_list

def calcAccuracy(W,b, testData, testTarget, errorList):
    test_data = testData.reshape((len(testData), len(testData[0])*len(testData[1])))
    y_hat = np.matmul(test_data, outW) + outb
    acc = 0
    for i in range(len(y_hat)):
        if(y_hat[i] >= 0.5):
            y_hat[i] = 1
        else:
            y_hat[i] = 0

    for i in range(len(testTarget)):
        if(y_hat[i] == testTarget[i]):
            acc+=1

    # printing the accuracy
    print(acc/len(testTarget))
    plt.plot(error)
    plt.ylabel('Error vs Epoch')
    plt.show()

def buildGraph(beta1=None, beta2=None, epsilon=None, lossType=None, learning_rate=None):
    # Your implementation here
