# -*- coding: utf-8 -*-
# %matplotlib inline   # for jupyter need
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# tf with linear regression
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.1 +0.3
# y_data = np.square(x_data) + 0.3


Weights = tf.Variable(tf.random_uniform([1], -1, 1))
biases = tf.Variable(tf.zeros([1]) + 0.1)
y =  Weights*x_data + biases

loss = tf.reduce_mean(tf.square(y_data - y))
train = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(201):
        sess.run(train)
        if i%20 ==0:
            print i, sess.run(Weights), sess.run(biases)
            plt.plot(x_data, y_data, 'ro', label="~")
            plt.plot(x_data, sess.run(Weights) * x_data + sess.run(biases), label = "****")
            plt.legend()
            plt.show()