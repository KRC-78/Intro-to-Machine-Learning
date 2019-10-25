import tensorflow as tf
import numpy as np
import os
import starter
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def CNN(epochs, batch_size, learning_rate, reg, dropout_prob):
    trainData, validData, testData, trainTarget, validTarget, testTarget = starter.loadData()
    trainData, validData, testData = np.expand_dims(trainData, 3), np.expand_dims(validData, 3), np.expand_dims(testData, 3)

    trainTarget, validTarget, testTarget = starter.convertOneHot(trainTarget, validTarget, testTarget)

    #Dimensions of 1 Sample
    x = tf.placeholder("float", [None, 28, 28, 1])

    #Dimensions of 1 Label (one-hot-encoded with 10 classes)
    target = tf.placeholder("float", [None, 10])

    #Filter & Filter Bias
    filt = tf.get_variable('filter', shape = (3, 3, 1, 32), initializer = tf.contrib.layers.xavier_initializer())
    filt_bias =  tf.get_variable('fitler_bias', shape = (32), initializer = tf.contrib.layers.xavier_initializer())

    #Weights and bias of First Fully Connected (Hidden) Layer
    W_h = tf.get_variable('W_h', shape = (32*14*14, 784), initializer = tf.contrib.layers.xavier_initializer())
    b_h = tf.get_variable('b_h', shape = (784), initializer = tf.contrib.layers.xavier_initializer())

    #Weights and bias of Second Fully Connected (Output) Layer
    W_o = tf.get_variable('W_o', shape = (784, 10), initializer = tf.contrib.layers.xavier_initializer())
    b_o = tf.get_variable('b_o', shape = (10), initializer = tf.contrib.layers.xavier_initializer())

    #2. Convolution Layer
    conv_layer = tf.nn.conv2d(x, filter=filt, strides=[1, 1, 1, 1], padding='SAME')
    conv_layer = tf.nn.bias_add(conv_layer, filt_bias)

    #3.ReLU activation for convolution layer
    relu_conv = tf.nn.relu(conv_layer)

    #Mean & variance calculations
    mean, variance = tf.nn.moments(relu_conv, axes=[0])

    #4. Batch nromalization layer
    #batch_norm_layer = tf.nn.batch_normalization(relu_conv, mean = mean, variance = variance)
    batch_norm_layer = tf.nn.batch_normalization(relu_conv, mean=mean, variance=variance, offset=None, scale=None, variance_epsilon=1e-8)

    #5. Max pooling layer
    max_pool_layer = tf.nn.max_pool(batch_norm_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    #6. Flattening Layer
    flat_layer = tf.reshape(max_pool_layer, [-1, W_h.get_shape().as_list()[0]])

    #7. Fully Connectec Hidden Layer
    fc_h = tf.add(tf.matmul(flat_layer, W_h), b_h)

    #For Section 2.3: Dropout layer
    dropout_layer = tf.nn.dropout(fc_h, keep_prob = (dropout_prob))

    #8.ReLU after hidden layer
    relu_h = tf.nn.relu(dropout_layer)
    #relu_h = tf.nn.relu(fc_h)

    #9. Fully connected output layer
    fc_o = tf.add(tf.matmul(relu_h, W_o), b_o)

    #10. Predictions are softmax activation of output layer(fc_o)
    predictions = tf.nn.softmax(fc_o)

    #Cross entropy loss function
    #Basic CE Loss
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=target, logits=predictions))

    #Loss with Regularization
    #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=target, logits=predictions)) + tf.multiply(tf.reduce_sum(tf.square(filt)) + tf.reduce_sum(tf.square(W_h)) + tf.reduce_sum(tf.square(W_o)), reg/2)

    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)

    pred_class = tf.argmax(predictions, 1)
    target_class = tf.argmax(target, 1)
    corr = tf.equal(pred_class, target_class)

    accuracy = tf.reduce_mean(tf.cast(corr, tf.float64))

    train_loss_array = []
    train_acc_array = []
    val_loss_array = []
    val_acc_array = []
    test_loss_array = []
    test_acc_array = []

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        N = trainData.shape[0]
        batches = int(N/ batch_size)

        for epoch in range(epochs):
            tot_train_loss = 0
            tot_train_corr = 0
            trainData, trainTarget = starter.shuffle(trainData, trainTarget)

            for batch in range(batches):
                start_idx = batch*batch_size
                end_idx = min((batch+1)*batch_size,N)

                feats = trainData[start_idx : end_idx]
                labels = trainTarget[start_idx : end_idx]

                sess.run(optimizer, feed_dict={x: feats, target: labels})

                train_loss, train_acc = sess.run([loss, accuracy], feed_dict={x: feats, target: labels})

                tot_train_loss += train_loss
                tot_train_corr += train_acc*feats.shape[0]
            epoch_train_loss, epoch_train_acc = (tot_train_loss/batches), (tot_train_corr/N)
            val_loss, val_acc = sess.run([loss, accuracy], feed_dict={x: validData, target: validTarget})
            test_loss, test_acc = sess.run([loss, accuracy], feed_dict={x: testData, target: testTarget})

            train_loss_array.append(epoch_train_loss)
            train_acc_array.append(epoch_train_acc)
            val_loss_array.append(val_loss)
            val_acc_array.append(val_acc)
            test_loss_array.append(test_loss)
            test_acc_array.append(test_acc)

            performance_data = [train_loss_array, train_acc_array,val_loss_array,val_acc_array,test_loss_array,test_acc_array]


    return performance_data

dp_5 = CNN(50, 32, 1e-4, 0, 0.5)
np.save("dp_5",dp_5)
