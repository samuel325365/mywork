# -*- coding: utf-8 -*-
# %matplotlib inline   # for jupyter need
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# tf with ^M regression

x_data = np.random.rand(100 ,1).astype(np.float32)
y_data = 2*x_data**3 - 5*np.square(x_data) + 0.3
data = tf.concat([x_data,x_data**2,x_data**3,x_data**4,x_data**5],1)

x_w = np.random.rand(300, 1).astype(np.float32)
data_w = tf.concat([x_w,x_w**2,x_w**3,x_w**4,x_w**5], 1)

Weights = tf.Variable(tf.random_uniform([5,1], -1, 1))
biases = tf.Variable(tf.zeros([1]) + 0.1)
y = tf.matmul(data, Weights) + biases

loss = tf.reduce_mean(tf.square(y_data - y))
train = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(201):
        sess.run(train)
        if i%20 ==0:
            print i, sess.run(Weights), sess.run(biases)
            plt.plot(x_data, y_data, 'ro', label="orgin_data")
            y = tf.add(tf.matmul(data_w,sess.run(Weights)),sess.run(biases)).eval()
            plt.plot(x_w, y, 'bo', label = "fun_of_W")
            plt.legend()
            plt.show()