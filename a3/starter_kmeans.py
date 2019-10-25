import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import helper as hlp

# Loading data
data = np.load('data2D.npy')
#data = np.load('data100D.npy')
[num_pts, dim] = np.shape(data)

# For Validation set
if is_valid:
  valid_batch = int(num_pts / 3.0)
  np.random.seed(45689)
  rnd_idx = np.arange(num_pts)
  np.random.shuffle(rnd_idx)
  val_data = data[rnd_idx[:valid_batch]]
  data = data[rnd_idx[valid_batch:]]


# Distance function for K-means
def distanceFunc(X, MU):
    # Inputs
    # X: is an NxD matrix (N observations and D dimensions)
    # MU: is an KxD matrix (K means and D dimensions)
    # Outputs
    # pair_dist: is the pairwise distance matrix (NxK)
    return tf.reduce_sum(tf.square(tf.expand_dims(X,1)-tf.expand_dims(MU,0)),2)

def buildGraph(learning_rate=None, dim=2, k=3):
    """
    :param learning_rate: optimizer's learning rate
    :return:
    """

    # Variable creation

    input_x = tf.placeholder(tf.float32, [None, dim], name='input_x')
    k_centers = tf.Variable(tf.random_normal([k, dim], stddev=0.5))
    data_size = tf.placeholder(tf.float32)
    distance = distanceFunc(input_x,k_centers)
    loss = (tf.reduce_sum(tf.reduce_min(distance, 1))) / data_size
    prediction = tf.argmin(distance,1)
    # Training mechanism
    if learning_rate is None:
        learning_rate = 0.01

    train = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.99,epsilon=1e-5).minimize(loss=loss)

    return input_x, loss, train, prediction,  k_centers, data_size
