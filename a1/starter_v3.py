import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import math


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

    grad_W = np.dot(X.T, (predictions - y)) / y.shape[0] + reg*W
    grad_b = (np.sum((predictions-y)))/ y.shape[0]
    return grad_W, grad_b

def grad_descent(W, b, trainingData, trainingLabels, alpha, iterations, reg, EPS, lossType="None"):
    # Your implementation here
    
    trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
    
    error = 2*EPS
    W = W.reshape((784,1))
    i=0
    e_list = []
    train_acc = []
    val_acc = []
    test_acc = []

    train_loss = []
    val_loss = []
    test_loss = []


    if lossType == "MSE":
        
        while(error > EPS and i<iterations):
            #print(MSE(W,b,trainingData,trainingLabels,reg))
            e_list.append(MSE(W,b,trainingData,trainingLabels,reg))
            gradW, gradb = gradMSE(W,b,trainingData,trainingLabels,reg)
            error = alpha*np.linalg.norm(gradW)
            W = np.subtract(W, alpha*gradW)
            b = b - alpha*gradb
            
            i+=1
    else:
        while(error > EPS and i<iterations):
            
            #Training Accuracy & Error
            s = np.dot(testData,W) + b
            predictions = 1 / (1 + np.exp(-s))
            error = alpha*np.linalg.norm(gradW)
            
            e_list.append(crossEntropyLoss(W,b,trainingData,trainingLabels,reg))
            gradW, gradb = gradCE(W,b,trainingData,trainingLabels,reg)
            W = np.subtract(W, alpha*gradW)
            b = b - alpha*gradb
            i+=1

    return W, b, e_list
    
'''trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()

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
print(acc/len(testTarget))'''



'''def buildGraph(epochs, learning_rate, batch_size, reg,beta_1 = "None", beta_2 = "None", epsil ="None",lossType="None"):
#def buildGraph(beta_1 = "None", beta_2 = "None", epsil ="None",lossType="None",):
    tf.set_random_seed(421)
    
    W = tf.Variable(tf.random.truncated_normal(shape = [784, 1],mean = 0,stddev = 0.5,dtype = tf.float32))
    b = tf.Variable(tf.random.truncated_normal(shape=[1, 1],mean=0,stddev=0.5,dtype=tf.float32))
    reg = tf.placeholder(tf.float32, name="reg")
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')

    raw_data = tf.placeholder(tf.float32, [None, 28, 28], name="input_")
    data = tf.reshape(raw_data, [-1, 28 * 28])
    labels = tf.placeholder(tf.float32, shape = [None], name="labels")
    
    #if lossType == "MSE":
    predictions = tf.add(tf.matmul(data, W),b, name = "MSE_predictions")
    mse = tf.reduce_mean(tf.square(predictions - labels)) / 2
    wd = tf.multiply(reg / 2, tf.reduce_sum(tf.square(W)))  # weight decay loss
    loss = mse + wd
        #loss = tf.losses.mean_squared_error(labels=tf.reshape(labels, [tf.shape(labels)[0],1]),predictions=predictions,) + tf.multiply(tf.reduce_sum(tf.square(W)), reg/2)
    #else:
        #s = tf.add(tf.matmul(data, W),b)
        #predictions = tf.math.sigmoid(-s,name = "CE_predictions")
        #loss = tf.losses.sigmoid_cross_entropy( multi_class_labels = tf.reshape(labels, [tf.shape(labels)[0],1]),logits = predictions) + tf.multiply(tf.reduce_sum(tf.square(W)), reg/2)

    #optimizer =  tf.train.AdamOptimizer(learning_rate=learning_rate, beta1 = beta_1, beta2 = beta_2, epsilon = epsil)
    optimizer =  tf.train.AdamOptimizer(learning_rate)#, beta1 = beta_1, beta2 = beta_2, epsilon = epsil)
    optimizer_step = optimizer.minimize(loss)
    
    accuracy = tf.reduce_mean(tf.to_float(tf.equal(tf.to_float(tf.greater(predictions, 0.5)), labels)))
    num_correct = tf.reduce_sum(tf.to_float(tf.equal(tf.to_float(tf.greater(predictions, 0.5)), labels)))

    trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
    
    init = tf.global_variables_initializer()
    error_list = []
    accuracy_list = []
    numBatches = math.ceil(len(trainTarget)/batch_size)

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(epochs):
            total_correct = 0
            
            total_train_correct = 0
            total_val_correct = 0
            total_test_correct = 0
            
            train_batch_loss = 0
            val_batch_loss = 0
            test_batch_loss = 0
            
            #Batch Shuffling
            permutation = np.random.permutation(trainTarget.shape[0])
            trainData = trainData[permutation]
            trainTarget = trainTarget[permutation]
            batch_loss = 0
    
            for i in range(0, numBatches):
                trainDataBatch = trainData[i:i+batch_size]
                trainTargetBatch = trainTarget[i:i+batch_size]
                
                sess.run(optimizer_step, feed_dict={raw_data: trainDataBatch, labels: trainTargetBatch, learning_rate: learning_rate, reg: reg})
                #sess.run(optimizer_adam, feed_dict={raw_x: trainDataBatch, y: trainTargetBatch, learning_rate: 0.001, lamb: 0})
                batch_loss += sess.run(loss, feed_dict={raw_data: trainDataBatch, labels: trainTargetBatch, learning_rate: learning_rate, reg: reg})
                #batch_accuracy = sess.run(accuracy, feed_dict={raw_x: trainDataBatch, y: trainTargetBatch, learning_rate: 0.001, lamb: 0})
                total_correct += sess.run(num_correct, feed_dict={raw_data: trainDataBatch, labels: trainTargetBatch, learning_rate: learning_rate, reg: reg})
    
            error_list.append(batch_loss)
            accuracy_list.append(total_correct/len(trainTarget))
            
    return W, b, raw_data, data, labels, loss, optimizer_step, reg, learning_rate, accuracy, num_correct
buildGraph(1000,0.001,500, 0,lossType = "MSE")

    trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
    W, b, raw_data, data, labels, loss, optimizer_step, reg, learning_rate, accuracy, num_correct =  buildGraph(lossType = lossType)#buildGraph(beta_1 = beta_1, beta_2 = beta_2, epsil = epsil,lossType=lossType )
    
    #train_acc = []
    #val_acc = []
    #test_acc = []

    #train_loss = []
    #val_loss = []
    #test_loss = []

    init = tf.global_variables_initializer()
    error_list = []
    accuracy_list = []
    numBatches = math.ceil(len(trainTarget)/batchSize)

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(epochs):
            total_correct = 0
            
            total_train_correct = 0
            total_val_correct = 0
            total_test_correct = 0
            
            train_batch_loss = 0
            val_batch_loss = 0
            test_batch_loss = 0
            
            #Batch Shuffling
            permutation = np.random.permutation(trainTarget.shape[0])
            trainData = trainData[permutation]
            trainTarget = trainTarget[permutation]
            batch_loss = 0
    
            for i in range(0, numBatches):
                trainDataBatch = trainData[i:i+batchSize]
                trainTargetBatch = trainTarget[i:i+batchSize]
                
                sess.run(optimizer_step, feed_dict={raw_data: trainDataBatch, labels: trainTargetBatch, learning_rate: learning_rate, reg: reg})
                batch_loss += sess.run(loss, feed_dict={raw_data: trainDataBatch, labels: trainTargetBatch, learning_rate: learning_rate, reg: reg})
                #batch_accuracy = sess.run(accuracy, feed_dict={raw_x: trainDataBatch, y: trainTargetBatch, learning_rate: 0.001, lamb: 0})
                total_correct += sess.run(num_correct, feed_dict={raw_data: trainDataBatch, labels: trainTargetBatch, learning_rate: learning_rate, reg: reg})
    
            error_list.append(batch_loss)
            accuracy_list.append(total_correct/len(trainTarget))
            
SGD(500,0.001,0,3000, lossType = "MSE")

#/***** Tensorflow with batching starts here ******/

'''
#def buildGraph(lossType = "None"):
def buildGraph(batch_size, lr, reg, epochs, beta_1 = "None", beta_2 = "None", epsil = "None",lossType = "None"):
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
        #pred_y = tf.sigmoid(pred_y)
        #ce = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.reshape(y, [tf.shape(y)[0],1]),logits = pred_y)) 
        ce = tf.losses.sigmoid_cross_entropy( multi_class_labels = tf.reshape(y, [tf.shape(y)[0],1]),logits = pred_y)
        loss = ce + wd
             
    #accuracy = tf.reduce_mean(tf.to_float(tf.equal(tf.to_float(tf.greater(pred_y, 0.5)), y)))
    num_correct = tf.reduce_sum(tf.to_float(tf.equal(tf.to_float(tf.greater(pred_y, 0.5)), y)))
    
    
    #optimizer_adam = tf.train.AdamOptimizer(learning_rate, beta1=beta_1, beta2 = beta_2, epsilon = epsil ).minimize(loss)
    optimizer_adam = tf.train.AdamOptimizer(learning_rate, epsilon=epsil).minimize(loss)
    init = tf.global_variables_initializer()

    train_acc = []
    val_acc = []
    test_acc = []

    train_loss = []
    val_loss = []
    test_loss = []


    numBatches = math.ceil(len(trainTarget)/batch_size)
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(epochs):
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


def plotter(tr_l, tr_a, v_l, v_a, te_l, te_a, bs,loss,beta1 = "None", beta2 = "None", epsil = "None"):
    loss = loss
    batch_size = bs
    
    alpha = 0.001
    reg = 0
    fig = plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(tr_l, label = 'Training Loss')
    plt.plot(v_l, label = 'Validation Loss')
    plt.plot(te_l, label = 'Test Loss')
    plt.title("{} Loss vs Epochs (Epsilon = {})".format(loss, beta1))
    #plt.title("{} Loss vs Epochs (Batch Size = {}, Learning Rate = {}, Reg={})".format(loss, batch_size, alpha, reg))
    plt.legend(loc = 0) 
    #plt.savefig("{} Loss vs Epochs (Batch Size = {}, Learning Rate = {}, Reg={}).png".format(loss, batch_size, alpha, reg))
    plt.savefig("{} Loss vs Epochs (Epsilon = {}).png".format(loss, beta1))
    plt.show()
    
    fig = plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.plot(scipy.signal.savgol_filter(tr_a, 7, 5), label = 'Training Accuracy')
    plt.plot(scipy.signal.savgol_filter(v_a, 7, 5), label = 'Validation Accuracy')
    plt.plot(scipy.signal.savgol_filter(te_a, 7, 5), label = 'Test Accuracy')
    plt.title("{} Accuracy vs Epochs (Epsilon = {})".format(loss, beta1))
    #plt.title("{} Accuracy vs Epochs (Batch Size = {})".format(loss, batch_size, alpha, reg))
    plt.legend(loc = 0)
    plt.ylim(0.0, 1.1)
    plt.savefig("{} Accuracy vs Epochs (Epsilon  = {}).png".format(loss, beta1))
    #plt.savefig("{} vs Epochs (Batch Size = {}, Learning Rate = {}, Reg={}).png".format(loss, batch_size, alpha, reg))
    plt.show()
    return
    
#bs_500_tr_l, bs_500_tr_a, bs_500_v_l, bs_500_v_a, bs_500_te_l, bs_500_te_a = buildGraph(500,0.001, 0,700, beta_1 = 0.95,lossType = "CE")
#bs_100_tr_l, bs_100_tr_a, bs_100_v_l, bs_100_v_a, bs_100_te_l, bs_100_te_a = buildGraph(100,0.001, 0,700, lossType = "CE")    
#bs_700_tr_l, bs_700_tr_a, bs_700_v_l, bs_700_v_a, bs_700_te_l, bs_700_te_a = buildGraph(700,0.001, 0,700, lossType = "CE")
#bs_1750_tr_l, bs_1750_tr_a, bs_1750_v_l, bs_1750_v_a, bs_1750_te_l, bs_1750_te_a = buildGraph(1750,0.001, 0,700, lossType = "CE")

#plotter(bs_500_tr_l, bs_500_tr_a, bs_500_v_l, bs_500_v_a, bs_500_te_l, bs_500_te_a,500)
#plotter(bs_100_tr_l, bs_100_tr_a, bs_100_v_l, bs_100_v_a, bs_100_te_l, bs_100_te_a,100)
#plotter(bs_700_tr_l, bs_700_tr_a, bs_700_v_l, bs_700_v_a, bs_700_te_l, bs_700_te_a,700)
#plotter(bs_1750_tr_l, bs_1750_tr_a, bs_1750_v_l, bs_1750_v_a, bs_1750_te_l, bs_1750_te_a,1750)

b1_95_CE_tr_l, b1_95_CE_tr_a, b1_95_CE_v_l, b1_95_CE_v_a, b1_95_CE_te_l, b1_95_CE_te_a = buildGraph(500,0.001, 0,700, epsil  = 1*(10**(-9)),lossType = "CE")
b1_99_CE_tr_l, b1_99_CE_tr_a, b1_99_CE_v_l, b1_99_CE_v_a, b1_99_CE_te_l, b1_99_CE_te_a = buildGraph(500,0.001, 0,700, epsil = 1*(10**(-4)),lossType = "CE")

b1_95_MSE_tr_l, b1_95_MSE_tr_a, b1_95_MSE_v_l, b1_95_MSE_v_a, b1_95_MSE_te_l, b1_95_MSE_te_a = buildGraph(500,0.001, 0,700, epsil = 1*(10**(-9)),lossType = "MSE")
b1_99_MSE_tr_l, b1_99_MSE_tr_a, b1_99_MSE_v_l, b1_99_MSE_v_a, b1_99_MSE_te_l, b1_99_MSE_te_a = buildGraph(500,0.001, 0,700, epsil = 1*(10**(-4)),lossType = "MSE")


plotter(b1_95_CE_tr_l, b1_95_CE_tr_a, b1_95_CE_v_l, b1_95_CE_v_a, b1_95_CE_te_l, b1_95_CE_te_a,500, "CE",beta1 = "1e-9")
plotter(b1_99_CE_tr_l, b1_99_CE_tr_a, b1_99_CE_v_l, b1_99_CE_v_a, b1_99_CE_te_l, b1_99_CE_te_a,500, "CE",beta1 = "1e-4")

plotter(b1_95_MSE_tr_l, b1_95_MSE_tr_a, b1_95_MSE_v_l, b1_95_MSE_v_a, b1_95_MSE_te_l, b1_95_MSE_te_a,500, "MSE",beta1 = "1e-9")
plotter(b1_99_MSE_tr_l, b1_99_MSE_tr_a, b1_99_MSE_v_l, b1_99_MSE_v_a, b1_99_MSE_te_l, b1_99_MSE_te_a,500, "MSE",beta1 = "1e-4")



