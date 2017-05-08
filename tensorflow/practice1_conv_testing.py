# -*- coding: utf-8 -*-
import tensorflow as tf
import os
import numpy as np
import visualized_analysis.reduce_dimension_function as va_rdf
current_dir = os.getcwd()

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

sess = tf.InteractiveSession()

xs = tf.placeholder(dtype=tf.float32, shape=[None, 784])
ys = tf.placeholder(dtype=tf.float32, shape=[None, 10])
x_image = tf.reshape(xs, shape=[-1, 28, 28, 1])
keep_prob = tf.placeholder(dtype=tf.float32)

#layer1
Weights_conv1 = tf.Variable(tf.truncated_normal(shape=[5,5,1,32], stddev=0.1), dtype=tf.float32, name='W_conv1')
biases_conv1 = tf.Variable(tf.constant(0.1, shape=[32]), dtype=tf.float32, name='b_conv1')
conv_1 = tf.nn.relu(tf.nn.conv2d(x_image, Weights_conv1, strides=[1,1,1,1], padding='SAME') + biases_conv1)
pool_layer1 = tf.nn.max_pool(conv_1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')  # outputs.shape = [n_samples, 14,14,32]

#layer2
Weights_conv2 = tf.Variable(tf.truncated_normal(shape=[5,5,32,48], stddev=0.1), dtype=tf.float32, name='W_conv2')
biases_conv2 = tf.Variable(tf.constant(0.1, shape=[48]), dtype=tf.float32, name='b_conv2')
conv_2 = tf.nn.relu(tf.nn.conv2d(pool_layer1, Weights_conv2, strides=[1,1,1,1], padding='SAME') + biases_conv2)  # outputs.shape = [n_samples, 14,14,48]
# pool_layer2 = tf.nn.max_pool(conv_2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

#layer3
Weights_conv3 = tf.Variable(tf.truncated_normal(shape=[5,5,48,64], stddev=0.1), dtype=tf.float32, name='W_conv3')
biases_conv3 = tf.Variable(tf.constant(0.1, shape=[64]), dtype=tf.float32, name='b_conv3')
conv_3 = tf.nn.relu(tf.nn.conv2d(conv_2, Weights_conv3, strides=[1,1,1,1], padding='SAME') + biases_conv3)
pool_layer3 = tf.nn.max_pool(conv_3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')  #outputs.shape = [n_samples, 7,7,64]

#fulled layer1
Weights_fc1 = tf.Variable(tf.truncated_normal(shape=[7*7*64, 1024], stddev=0.1), dtype=tf.float32, name='W_fc1')
biases_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]), dtype=tf.float32, name='b_fc1')
fc1 = tf.reshape(pool_layer3, shape=[-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(fc1, Weights_fc1) + biases_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#fulled layer2
Weights_fc2 = tf.Variable(tf.truncated_normal(shape=[1024, 10], stddev=0.1), dtype=tf.float32, name='W_fc2')
biases_fc2 = tf .Variable(tf.constant(0.1, shape=[10]), dtype=tf.float32, name='b_fc2')
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, Weights_fc2) + biases_fc2)

saver = tf.train.Saver()
saver.restore(sess, os.path.join(current_dir, 'models/practice1_conv/model.ckpt'))

test_size = 5000
test_data = mnist.test.images[0:test_size, :]
test_label = mnist.test.labels[0:test_size, :]
test_label_index = np.argmax(test_label, axis=1)


# PCA
layer3_output_reshape = tf.reshape(pool_layer3[:,:,:,:], shape=[-1, 7*7*64])
test_layer3_pca = va_rdf.pca(layer3_output_reshape.eval(feed_dict={xs:test_data}), 2)
va_rdf.plot_scatter(test_layer3_pca, test_label_index, "conv layer3 with pca", txt=True)

# TSNE
fc1_pca = va_rdf.pca(h_fc1.eval(feed_dict ={ xs: test_data, keep_prob:1}), 50)
fc1_tsne = va_rdf.tsne(fc1_pca, 2)
va_rdf.plot_scatter(fc1_tsne, test_label_index, "fc layer1 with tsne", txt=True)
