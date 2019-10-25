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
    X = np.reshape(x, (x.shape[0], -1))
    s = np.dot(X,W) + b
    predictions = 1 / (1 + np.exp(-s))


    loss_d = -np.dot(y.T,np.log(predictions)) - np.dot((1-y).T,np.log((1-predictions)))
    loss_d = loss_d/y.shape[0]

    loss_w = np.dot(W.transpose(),W)
    loss_w = (reg/2) * loss_w

    loss = loss_d + loss_w

    return loss

def gradCE(W, b, x, y, reg):
    X = np.reshape(x, (x.shape[0], -1))
    s = np.dot(X,W) + b
    predictions = 1 / (1 + np.exp(-s))

    grad_W = np.dot(X.T, (predictions - y)) / y.shape[0]
    grad_b = (np.sum((predictions-y)))/ y.shape[0]
    return grad_W, grad_b

def grad_descent(W, b, trainingData, trainingLabels, alpha, iterations, reg, EPS, lossType="None"):
    # Your implementation here
    error = 2*EPS
    W = W.reshape((784,1))
    i=0
    e_list = []
    train_acc = []
    val_acc = []
    test_acc = []

    train_error = []
    val_error = []
    test_error = []


    if lossType == "MSE":

        while(error > EPS and i<iterations):
            #print(MSE(W,b,trainingData,trainingLabels,reg))
            e_list.append(MSE(W,b,trainingData,trainingLabels,reg))
            gradW, gradb = gradMSE(W,b,trainingData,trainingLabels,reg)
            error = alpha*np.linalg.norm(gradW)
            W = np.subtract(W, alpha*gradW)
            b = b - alpha*gradb

            #error = MSE(W,b,trainingData,trainingLabels,reg)
            i+=1
    else:
        while(error > EPS and i<iterations):

            #Training Accuracy & Error
            s = np.dot(testData,W) + b
            predictions = 1 / (1 + np.exp(-s))


            e_list.append(crossEntropyLoss(W,b,trainingData,trainingLabels,reg))
            gradW, gradb = gradCE(W,b,trainingData,trainingLabels,reg)
            W = np.subtract(W, alpha*gradW)
            b = b - alpha*gradb
            i+=1

    return W, b, e_list

trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()

W = np.zeros((784,1))
b = 0
acc = 0
W,b, e_list = grad_descent(W, b, trainData, trainTarget, 0.01, 5000, 0.1, 0.0000001, lossType="CE")
testData = np.reshape(testData, (testData.shape[0], -1))
s = np.dot(testData,W) + b
y_hat = 1 / (1 + np.exp(-s))
for i in range(len(testTarget)):
    if((y_hat[i] > 0.5 and testTarget[i] == 1) or (y_hat[i] < 0.5 and testTarget[i] == 0)):
        acc +=1
print(acc/len(testTarget))

def buildGraph(batch_size, lr, reg, lossType = "None"):
    #batchSize = 500
    trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
    
    raw_x = tf.placeholder(tf.float32, [None, 28, 28], name="input_x")
    x = tf.reshape(raw_x, [-1, 28 * 28])
    y = tf.placeholder(tf.float32, [None, 1], name="input_y")
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    lamb = tf.placeholder(tf.float32, name='lambda')
    #w = tf.Variable(tf.zeros([28 * 28, 1], dtype=tf.float32))
    W = tf.Variable(tf.random.truncated_normal(shape = [784, 1],mean = 0,stddev = 0.5,dtype = tf.float32))
    b = tf.Variable(tf.random.truncated_normal(shape=[1, 1],mean=0,stddev=0.5,dtype=tf.float32))
    
    pred_y = tf.matmul(x, W) + b
    wd = tf.multiply(lamb / 2, tf.reduce_sum(tf.square(W)))
    
    if lossType == "MSE":
        mse = tf.reduce_mean(tf.square(pred_y - y)) / 2
        loss = mse + wd
    else:
        pred_y = tf.sigmoid(pred_y, name = "predictionsSig")
        ce = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.reshape(y, [tf.shape(y)[0],1]),logits = pred_y))
        loss = ce + wd
        

        
    #accuracy = tf.reduce_mean(tf.to_float(tf.equal(tf.to_float(tf.greater(pred_y, 0.5)), y)))
    num_correct = tf.reduce_sum(tf.to_float(tf.equal(tf.to_float(tf.greater(pred_y, 0.5)), y)))
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    optimizer_adam = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    
    init = tf.global_variables_initializer()
    #error_list = []
    #accuracy_list = []

    train_acc = []
    val_acc = []
    test_acc = []

    train_loss = []
    val_loss = []
    test_loss = []



    numBatches = math.ceil(len(trainTarget)/batch_size)
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(3000):
            total_correct = 0
            permutation = np.random.permutation(trainTarget.shape[0])
            trainData = trainData[permutation]
            trainTarget = trainTarget[permutation]
            batch_loss = 0
    
            for i in range(0, numBatches):
                trainDataBatch = trainData[i:i+batch_size]
                trainTargetBatch = trainTarget[i:i+batch_size]
                sess.run(optimizer_adam, feed_dict={raw_x: trainDataBatch, y: trainTargetBatch, learning_rate: lr, lamb: reg})
                batch_loss += sess.run(loss, feed_dict={raw_x: trainDataBatch, y: trainTargetBatch, learning_rate: lr, lamb: reg})
                #batch_accuracy = sess.run(accuracy, feed_dict={raw_x: trainDataBatch, y: trainTargetBatch, learning_rate: 0.001, lamb: 0})
                total_correct += sess.run(num_correct, feed_dict={raw_x: trainDataBatch, y: trainTargetBatch, learning_rate: lr, lamb: reg})
            
            #Training
            train_loss.append(batch_loss)
            train_acc.append(total_correct/len(trainTarget))
            
            #Validation
            val_corr = sess.run(num_correct, feed_dict={raw_x: validData, y: validTarget, learning_rate: lr, lamb: reg})
            cur_val_loss = sess.run(loss, feed_dict={raw_x: validData, y: validTarget, learning_rate: lr, lamb: reg})
            
            val_loss.append(cur_val_loss)
            val_acc.append(val_corr/len(validTarget))
            
            #Test
            test_corr = sess.run(num_correct, feed_dict={raw_x: testData, y: testTarget, learning_rate: lr, lamb: reg})
            cur_test_loss = sess.run(loss, feed_dict={raw_x: testData, y: testTarget, learning_rate: lr, lamb: reg})
            
            test_loss.append(cur_test_loss)
            test_acc.append(test_corr/len(testTarget))
            
    return train_loss, train_acc, val_loss, val_acc, test_loss, test_acc
#tr_l, tr_a, v_l, v_a, te_l, te_a = buildGraph(700,0.001, 0, lossType = "None")

